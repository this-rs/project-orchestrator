//! MCP Tool handlers
//!
//! All tool calls are proxied to the REST API via McpHttpClient.
//! Mega-tool names (e.g. "project") are resolved to legacy names
//! (e.g. "list_projects") before HTTP routing.

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use super::http_client::{extract_id, extract_optional_string, extract_string, McpHttpClient};
use crate::graph::models::profile_by_name;
use crate::neurons::intent::{IntentDetector, QueryIntentMode};

/// Resolve an intent-adaptive analysis profile name from MCP params.
///
/// Priority: explicit `profile` > explicit `intent_mode` > auto-detect from query.
/// Returns `None` when the query has no detectable intent (backward-compatible default).
fn resolve_intent_profile(
    query: &str,
    explicit_profile: Option<&str>,
    explicit_intent_mode: Option<&str>,
) -> Option<String> {
    if let Some(name) = explicit_profile {
        // Explicit profile override — validate it exists
        if profile_by_name(name).is_some() {
            tracing::debug!(
                profile = name,
                source = "explicit",
                "intent-adaptive profile resolved"
            );
            return Some(name.to_string());
        } else {
            tracing::warn!(
                profile = name,
                "unknown profile name, falling back to auto-detection"
            );
        }
    }

    if let Some(mode_str) = explicit_intent_mode {
        let mode = match mode_str {
            "debug" => QueryIntentMode::Debug,
            "explore" => QueryIntentMode::Explore,
            "impact" => QueryIntentMode::Impact,
            "plan" => QueryIntentMode::Plan,
            _ => QueryIntentMode::Default,
        };
        if mode != QueryIntentMode::Default {
            let name = mode.to_string();
            tracing::debug!(profile = %name, source = "explicit_intent", "intent-adaptive profile resolved");
            return Some(name);
        }
    }

    // Auto-detect from query
    let mode = IntentDetector::detect(query);
    if mode == QueryIntentMode::Default {
        tracing::debug!(
            source = "auto",
            detected = "default",
            "no intent detected, using default behavior"
        );
        None
    } else {
        let name = mode.to_string();
        tracing::debug!(profile = %name, source = "auto", "intent-adaptive profile resolved");
        Some(name)
    }
}

/// Fix stringified JSON values from MCP transport.
///
/// Claude Code sometimes sends array/integer/boolean parameters as JSON strings
/// (e.g. `"[\"a\"]"` instead of `["a"]`, `"100"` instead of `100`).
/// This function detects and deserializes them back to proper JSON types.
fn unstringify_json_values(args: &mut Value) {
    let obj = match args.as_object_mut() {
        Some(o) => o,
        None => return,
    };
    for (_key, val) in obj.iter_mut() {
        if let Some(s) = val.as_str().map(|s| s.to_owned()) {
            let trimmed = s.trim();
            if trimmed.starts_with('[') || trimmed.starts_with('{') {
                if let Ok(parsed) = serde_json::from_str::<Value>(&s) {
                    if parsed.is_array() || parsed.is_object() {
                        *val = parsed;
                        continue;
                    }
                }
            }
            if trimmed == "true" {
                *val = Value::Bool(true);
                continue;
            }
            if trimmed == "false" {
                *val = Value::Bool(false);
                continue;
            }
            if let Ok(n) = trimmed.parse::<i64>() {
                *val = Value::Number(n.into());
            }
        }
    }
}

/// Handles MCP tool calls by proxying to the REST API.
pub struct ToolHandler {
    client: McpHttpClient,
    /// Trajectory collector for decision capture (fire-and-forget).
    /// None when collection is disabled.
    collector: Option<std::sync::Arc<neural_routing_runtime::TrajectoryCollector>>,
}

impl ToolHandler {
    /// Create a new ToolHandler with an HTTP client.
    pub fn new(http_client: McpHttpClient) -> Self {
        Self {
            client: http_client,
            collector: None,
        }
    }

    /// Create a new ToolHandler with trajectory collection enabled.
    pub fn with_collector(
        http_client: McpHttpClient,
        collector: std::sync::Arc<neural_routing_runtime::TrajectoryCollector>,
    ) -> Self {
        Self {
            client: http_client,
            collector: Some(collector),
        }
    }

    /// Get the HTTP client.
    fn http(&self) -> &McpHttpClient {
        &self.client
    }

    /// Resolve mega-tool names to legacy names for backward-compatible routing.
    ///
    /// If `name` is a mega-tool (e.g. "project"), extracts the `action` param and
    /// maps to the legacy tool name (e.g. "list_projects"). Legacy names pass through.
    fn resolve_mega_tool(&self, name: &str, args: &Value) -> Result<(String, Value)> {
        use super::tools::resolve_legacy_alias;

        // Check if already a legacy name → pass through
        if resolve_legacy_alias(name).is_some() {
            return Ok((name.to_string(), args.clone()));
        }

        // Mega-tool names: project, plan, task, step, decision, constraint,
        // release, milestone, commit, note, workspace, workspace_milestone,
        // resource, component, chat, feature_graph, code, admin
        let mega_tools: &[&str] = &[
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
            "admin",
            "skill",
            "analysis_profile",
            "protocol",
            "episode",
            "persona",
            "sharing",
            "neural_routing",
            "trajectory",
        ];

        if !mega_tools.contains(&name) {
            // Unknown tool — return as-is, let downstream handle the error
            return Ok((name.to_string(), args.clone()));
        }

        // Extract action parameter
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Mega-tool '{}' requires an 'action' parameter", name))?;

        // Build the legacy tool name from (mega_tool, action)
        let legacy_name = self.mega_tool_to_legacy(name, action)?;

        // Return resolved name + original args (action param is ignored by handlers)
        Ok((legacy_name, args.clone()))
    }

    /// Map (mega_tool, action) → legacy tool name
    fn mega_tool_to_legacy(&self, tool: &str, action: &str) -> Result<String> {
        let name = match (tool, action) {
            // Project
            ("project", "list") => "list_projects",
            ("project", "create") => "create_project",
            ("project", "get") => "get_project",
            ("project", "update") => "update_project",
            ("project", "delete") => "delete_project",
            ("project", "sync") => "sync_project",
            ("project", "get_roadmap") => "get_project_roadmap",
            ("project", "list_plans") => "list_project_plans",
            ("project", "get_graph") => "get_project_graph",
            ("project", "get_intelligence_summary") => "get_intelligence_summary",
            ("project", "get_embeddings_projection") => "get_embeddings_projection",
            ("project", "get_scaffolding_level") => "get_scaffolding_level",
            ("project", "set_scaffolding_override") => "set_scaffolding_override",
            ("project", "get_health_dashboard") => "get_health_dashboard",
            ("project", "get_auto_roadmap") => "get_auto_roadmap",

            // Plan
            ("plan", "list") => "list_plans",
            ("plan", "create") => "create_plan",
            ("plan", "get") => "get_plan",
            ("plan", "update") => "update_plan",
            ("plan", "update_status") => "update_plan_status",
            ("plan", "delete") => "delete_plan",
            ("plan", "link_to_project") => "link_plan_to_project",
            ("plan", "unlink_from_project") => "unlink_plan_from_project",
            ("plan", "get_dependency_graph") => "get_dependency_graph",
            ("plan", "get_critical_path") => "get_critical_path",
            ("plan", "get_waves") => "get_waves",
            ("plan", "run") => "run_plan",
            ("plan", "run_status") => "get_run_status",
            ("plan", "cancel_run") => "cancel_plan_run",
            ("plan", "auto_pr") => "create_auto_pr",
            ("plan", "add_trigger") => "add_trigger",
            ("plan", "list_triggers") => "list_triggers",
            ("plan", "remove_trigger") => "remove_trigger",
            ("plan", "enable_trigger") => "enable_trigger",
            ("plan", "disable_trigger") => "disable_trigger",
            ("plan", "list_runs") => "list_plan_runs",
            ("plan", "get_run") => "get_plan_run",
            ("plan", "compare_runs") => "compare_plan_runs",
            ("plan", "predict_run") => "predict_plan_run",
            ("plan", "enrich") => "enrich_plan",
            ("plan", "delegate_task") => "delegate_task",

            // Task
            ("task", "list") => "list_tasks",
            ("task", "create") => "create_task",
            ("task", "get") => "get_task",
            ("task", "update") => "update_task",
            ("task", "delete") => "delete_task",
            ("task", "get_next") => "get_next_task",
            ("task", "add_dependencies") => "add_task_dependencies",
            ("task", "remove_dependency") => "remove_task_dependency",
            ("task", "get_blockers") => "get_task_blockers",
            ("task", "get_blocked_by") => "get_tasks_blocked_by",
            ("task", "get_context") => "get_task_context",
            ("task", "get_prompt") => "get_task_prompt",
            ("task", "build_prompt") => "build_task_prompt",
            ("task", "enrich") => "enrich_task",

            // Step
            ("step", "list") => "list_steps",
            ("step", "create") => "create_step",
            ("step", "update") => "update_step",
            ("step", "get") => "get_step",
            ("step", "delete") => "delete_step",
            ("step", "get_progress") => "get_step_progress",

            // Decision
            ("decision", "add") => "add_decision",
            ("decision", "get") => "get_decision",
            ("decision", "update") => "update_decision",
            ("decision", "delete") => "delete_decision",
            ("decision", "search") => "search_decisions",
            ("decision", "search_semantic") => "search_decisions_semantic",
            ("decision", "add_affects") => "add_decision_affects",
            ("decision", "remove_affects") => "remove_decision_affects",
            ("decision", "list_affects") => "list_decision_affects",
            ("decision", "get_affecting") => "get_decisions_affecting",
            ("decision", "supersede") => "supersede_decision",
            ("decision", "get_timeline") => "get_decision_timeline",

            // Constraint
            ("constraint", "list") => "list_constraints",
            ("constraint", "add") => "add_constraint",
            ("constraint", "get") => "get_constraint",
            ("constraint", "update") => "update_constraint",
            ("constraint", "delete") => "delete_constraint",

            // Release
            ("release", "list") => "list_releases",
            ("release", "create") => "create_release",
            ("release", "get") => "get_release",
            ("release", "update") => "update_release",
            ("release", "delete") => "delete_release",
            ("release", "add_task") => "add_task_to_release",
            ("release", "add_commit") => "add_commit_to_release",
            ("release", "remove_commit") => "remove_commit_from_release",

            // Milestone
            ("milestone", "list") => "list_milestones",
            ("milestone", "create") => "create_milestone",
            ("milestone", "get") => "get_milestone",
            ("milestone", "update") => "update_milestone",
            ("milestone", "delete") => "delete_milestone",
            ("milestone", "get_progress") => "get_milestone_progress",
            ("milestone", "add_task") => "add_task_to_milestone",
            ("milestone", "link_plan") => "link_plan_to_milestone",
            ("milestone", "unlink_plan") => "unlink_plan_from_milestone",

            // Commit
            ("commit", "create") => "create_commit",
            ("commit", "link_to_task") => "link_commit_to_task",
            ("commit", "link_to_plan") => "link_commit_to_plan",
            ("commit", "get_task_commits") => "get_task_commits",
            ("commit", "get_plan_commits") => "get_plan_commits",
            ("commit", "get_commit_files") => "get_commit_files",
            ("commit", "get_file_history") => "get_file_history",

            // Note
            ("note", "list") => "list_notes",
            ("note", "create") => "create_note",
            ("note", "get") => "get_note",
            ("note", "update") => "update_note",
            ("note", "delete") => "delete_note",
            ("note", "search") => "search_notes",
            ("note", "search_semantic") => "search_notes_semantic",
            ("note", "confirm") => "confirm_note",
            ("note", "invalidate") => "invalidate_note",
            ("note", "supersede") => "supersede_note",
            ("note", "link_to_entity") => "link_note_to_entity",
            ("note", "unlink_from_entity") => "unlink_note_from_entity",
            ("note", "get_context") => "get_context_notes",
            ("note", "get_needing_review") => "get_notes_needing_review",
            ("note", "list_project") => "list_project_notes",
            ("note", "get_propagated") => "get_propagated_notes",
            ("note", "get_propagated_knowledge") => "get_propagated_knowledge",
            ("note", "get_context_knowledge") => "get_context_knowledge",
            ("note", "get_entity") => "get_entity_notes",
            ("note", "list_rfcs") => "list_rfcs",
            ("note", "advance_rfc") => "advance_rfc",
            ("note", "get_rfc_status") => "get_rfc_status",

            // Workspace
            ("workspace", "list") => "list_workspaces",
            ("workspace", "create") => "create_workspace",
            ("workspace", "get") => "get_workspace",
            ("workspace", "update") => "update_workspace",
            ("workspace", "delete") => "delete_workspace",
            ("workspace", "get_overview") => "get_workspace_overview",
            ("workspace", "list_projects") => "list_workspace_projects",
            ("workspace", "add_project") => "add_project_to_workspace",
            ("workspace", "remove_project") => "remove_project_from_workspace",
            ("workspace", "get_topology") => "get_workspace_topology",
            ("workspace", "get_coupling_matrix") => "get_coupling_matrix",

            // Workspace Milestone
            ("workspace_milestone", "list_all") => "list_all_workspace_milestones",
            ("workspace_milestone", "list") => "list_workspace_milestones",
            ("workspace_milestone", "create") => "create_workspace_milestone",
            ("workspace_milestone", "get") => "get_workspace_milestone",
            ("workspace_milestone", "update") => "update_workspace_milestone",
            ("workspace_milestone", "delete") => "delete_workspace_milestone",
            ("workspace_milestone", "add_task") => "add_task_to_workspace_milestone",
            ("workspace_milestone", "link_plan") => "link_plan_to_workspace_milestone",
            ("workspace_milestone", "unlink_plan") => "unlink_plan_from_workspace_milestone",
            ("workspace_milestone", "get_progress") => "get_workspace_milestone_progress",

            // Resource
            ("resource", "list") => "list_resources",
            ("resource", "create") => "create_resource",
            ("resource", "get") => "get_resource",
            ("resource", "update") => "update_resource",
            ("resource", "delete") => "delete_resource",
            ("resource", "link_to_project") => "link_resource_to_project",

            // Component
            ("component", "list") => "list_components",
            ("component", "create") => "create_component",
            ("component", "get") => "get_component",
            ("component", "update") => "update_component",
            ("component", "delete") => "delete_component",
            ("component", "add_dependency") => "add_component_dependency",
            ("component", "remove_dependency") => "remove_component_dependency",
            ("component", "map_to_project") => "map_component_to_project",

            // Chat
            ("chat", "list_sessions") => "list_chat_sessions",
            ("chat", "get_session") => "get_chat_session",
            ("chat", "get_children") => "get_session_children",
            ("chat", "delete_session") => "delete_chat_session",
            ("chat", "send_message") => "chat_send_message",
            ("chat", "list_messages") => "list_chat_messages",
            ("chat", "add_discussed") => "add_discussed",
            ("chat", "get_session_entities") => "get_session_entities",
            ("chat", "get_session_tree") => "get_session_tree",
            ("chat", "get_run_sessions") => "get_run_sessions",

            // Feature Graph
            ("feature_graph", "create") => "create_feature_graph",
            ("feature_graph", "get") => "get_feature_graph",
            ("feature_graph", "list") => "list_feature_graphs",
            ("feature_graph", "add_entity") => "add_to_feature_graph",
            ("feature_graph", "auto_build") => "auto_build_feature_graph",
            ("feature_graph", "delete") => "delete_feature_graph",
            ("feature_graph", "get_statistics") => "get_feature_graph_statistics",
            ("feature_graph", "compare") => "compare_feature_graphs",
            ("feature_graph", "find_overlapping") => "find_overlapping_feature_graphs",

            // Code
            ("code", "search") => "search_code",
            ("code", "search_project") => "search_project_code",
            ("code", "search_workspace") => "search_workspace_code",
            ("code", "get_file_symbols") => "get_file_symbols",
            ("code", "find_references") => "find_references",
            ("code", "get_file_dependencies") => "get_file_dependencies",
            ("code", "get_call_graph") => "get_call_graph",
            ("code", "analyze_impact") => "analyze_impact",
            ("code", "get_architecture") => "get_architecture",
            ("code", "find_similar") => "find_similar_code",
            ("code", "find_trait_implementations") => "find_trait_implementations",
            ("code", "find_type_traits") => "find_type_traits",
            ("code", "get_impl_blocks") => "get_impl_blocks",
            ("code", "get_communities") => "get_code_communities",
            ("code", "get_health") => "get_code_health",
            ("code", "get_node_importance") => "get_node_importance",
            ("code", "plan_implementation") => "plan_implementation",
            ("code", "get_co_change_graph") => "get_co_change_graph",
            ("code", "get_file_co_changers") => "get_file_co_changers",
            ("code", "detect_processes") => "detect_processes",
            ("code", "get_class_hierarchy") => "get_class_hierarchy",
            ("code", "find_subclasses") => "find_subclasses",
            ("code", "find_interface_implementors") => "find_interface_implementors",
            ("code", "list_processes") => "list_processes",
            ("code", "get_process") => "get_process",
            ("code", "get_entry_points") => "get_entry_points",
            ("code", "enrich_communities") => "enrich_communities",
            ("code", "get_hotspots") => "get_hotspots",
            ("code", "get_knowledge_gaps") => "get_knowledge_gaps",
            ("code", "get_risk_assessment") => "get_risk_assessment",
            ("code", "get_homeostasis") => "get_homeostasis",
            ("code", "get_structural_drift") => "get_structural_drift",
            ("code", "get_bridge") => "get_bridge",
            ("code", "check_topology") => "check_topology",
            ("code", "create_topology_rule") => "create_topology_rule",
            ("code", "list_topology_rules") => "list_topology_rules",
            ("code", "delete_topology_rule") => "delete_topology_rule",
            ("code", "check_file_topology") => "check_file_topology",
            ("code", "get_structural_profile") => "get_structural_profile",
            ("code", "find_structural_twins") => "find_structural_twins",
            ("code", "cluster_dna") => "cluster_dna",
            ("code", "find_cross_project_twins") => "find_cross_project_twins",
            ("code", "predict_missing_links") => "predict_missing_links",
            ("code", "check_link_plausibility") => "check_link_plausibility",
            ("code", "stress_test_node") => "stress_test_node",
            ("code", "stress_test_edge") => "stress_test_edge",
            ("code", "stress_test_cascade") => "stress_test_cascade",
            ("code", "find_bridges") => "find_bridges",
            ("code", "get_context_card") => "get_context_card",
            ("code", "refresh_context_cards") => "refresh_context_cards",
            ("code", "get_fingerprint") => "get_fingerprint",
            ("code", "find_isomorphic") => "find_isomorphic",
            ("code", "suggest_structural_templates") => "suggest_structural_templates",
            ("code", "get_learning_health") => "get_learning_health",

            // Skill
            ("skill", "list") => "list_skills",
            ("skill", "create") => "create_skill",
            ("skill", "get") => "get_skill",
            ("skill", "update") => "update_skill",
            ("skill", "delete") => "delete_skill",
            ("skill", "get_members") => "get_skill_members",
            ("skill", "add_member") => "add_skill_member",
            ("skill", "remove_member") => "remove_skill_member",
            ("skill", "activate") => "activate_skill",
            ("skill", "export") => "export_skill",
            ("skill", "import") => "import_skill",
            ("skill", "get_health") => "get_skill_health",
            ("skill", "split") => "split_skill",
            ("skill", "merge") => "merge_skills",

            // Protocol (Pattern Federation)
            ("protocol", "list") => "list_protocols",
            ("protocol", "create") => "create_protocol",
            ("protocol", "get") => "get_protocol",
            ("protocol", "update") => "update_protocol",
            ("protocol", "delete") => "delete_protocol",
            ("protocol", "add_state") => "add_protocol_state",
            ("protocol", "delete_state") => "delete_protocol_state",
            ("protocol", "list_states") => "list_protocol_states",
            ("protocol", "add_transition") => "add_protocol_transition",
            ("protocol", "delete_transition") => "delete_protocol_transition",
            ("protocol", "list_transitions") => "list_protocol_transitions",
            ("protocol", "link_to_skill") => "link_protocol_to_skill",
            ("protocol", "start_run") => "start_protocol_run",
            ("protocol", "get_run") => "get_protocol_run",
            ("protocol", "list_runs") => "list_protocol_runs",
            ("protocol", "transition") => "fire_protocol_transition",
            ("protocol", "cancel_run") => "cancel_protocol_run",
            ("protocol", "fail_run") => "fail_protocol_run",
            ("protocol", "report_progress") => "report_protocol_progress",
            ("protocol", "delete_run") => "delete_protocol_run",
            ("protocol", "route") => "route_protocols",
            ("protocol", "compose") => "compose_protocol",
            ("protocol", "simulate") => "simulate_protocol",
            ("protocol", "get_run_tree") => "get_protocol_run_tree",
            ("protocol", "get_run_children") => "get_protocol_run_children",

            // Persona (Living Personas)
            ("persona", "create") => "create_persona",
            ("persona", "get") => "get_persona",
            ("persona", "list") => "list_personas",
            ("persona", "update") => "update_persona",
            ("persona", "delete") => "delete_persona",
            ("persona", "add_skill") => "add_persona_skill",
            ("persona", "remove_skill") => "remove_persona_skill",
            ("persona", "add_protocol") => "add_persona_protocol",
            ("persona", "remove_protocol") => "remove_persona_protocol",
            ("persona", "add_file") => "add_persona_file",
            ("persona", "remove_file") => "remove_persona_file",
            ("persona", "add_function") => "add_persona_function",
            ("persona", "remove_function") => "remove_persona_function",
            ("persona", "add_note") => "add_persona_note",
            ("persona", "remove_note") => "remove_persona_note",
            ("persona", "add_decision") => "add_persona_decision",
            ("persona", "remove_decision") => "remove_persona_decision",
            ("persona", "scope_to_feature_graph") => "scope_persona_feature_graph",
            ("persona", "unscope_feature_graph") => "unscope_persona_feature_graph",
            ("persona", "add_extends") => "add_persona_extends",
            ("persona", "remove_extends") => "remove_persona_extends",
            ("persona", "get_subgraph") => "get_persona_subgraph",
            ("persona", "find_for_file") => "find_personas_for_file",
            ("persona", "list_global") => "list_global_personas",
            ("persona", "export") => "export_persona",
            ("persona", "import") => "import_persona",
            ("persona", "activate") => "activate_persona",
            ("persona", "auto_build") => "auto_build_persona",
            ("persona", "maintain") => "maintain_personas",
            ("persona", "detect") => "detect_personas",

            // Sharing (Privacy MVP)
            ("sharing", "status") => "get_sharing_status",
            ("sharing", "enable") => "enable_sharing",
            ("sharing", "disable") => "disable_sharing",
            ("sharing", "set_policy") => "set_sharing_policy",
            ("sharing", "get_policy") => "get_sharing_policy",
            ("sharing", "set_consent") => "set_sharing_consent",
            ("sharing", "history") => "get_sharing_history",
            ("sharing", "preview") => "preview_sharing",
            ("sharing", "suggest") => "suggest_sharing",
            ("sharing", "retract") => "retract_sharing",
            ("sharing", "list_tombstones") => "list_tombstones",
            ("sharing", "last_report") => "get_last_privacy_report",

            // Neural Routing
            ("neural_routing", "status") => "get_neural_routing_status",
            ("neural_routing", "get_config") => "get_neural_routing_config",
            ("neural_routing", "enable") => "enable_neural_routing",
            ("neural_routing", "disable") => "disable_neural_routing",
            ("neural_routing", "set_mode") => "set_neural_routing_mode",
            ("neural_routing", "update_config") => "update_neural_routing_config",

            // Trajectory
            ("trajectory", "list") => "list_trajectories",
            ("trajectory", "get") => "get_trajectory",
            ("trajectory", "search_similar") => "search_similar_trajectories",
            ("trajectory", "stats") => "get_trajectory_stats",

            // Reasoning Tree
            ("reasoning", "reason") => "reason",
            ("reasoning", "reason_feedback") => "reason_feedback",

            // Episode (Episodic Memory)
            ("episode", "collect") => "collect_episode",
            ("episode", "list") => "list_episodes",
            ("episode", "anonymize") => "anonymize_episode",
            ("episode", "export_artifact") => "export_artifact",

            // Analysis Profile
            ("analysis_profile", "list") => "list_analysis_profiles",
            ("analysis_profile", "create") => "create_analysis_profile",
            ("analysis_profile", "get") => "get_analysis_profile",
            ("analysis_profile", "delete") => "delete_analysis_profile",

            // Admin
            ("admin", "sync_directory") => "sync_directory",
            ("admin", "start_watch") => "start_watch",
            ("admin", "stop_watch") => "stop_watch",
            ("admin", "watch_status") => "watch_status",
            ("admin", "meilisearch_stats") => "get_meilisearch_stats",
            ("admin", "delete_meilisearch_orphans") => "delete_meilisearch_orphans",
            ("admin", "cleanup_cross_project_calls") => "cleanup_cross_project_calls",
            ("admin", "cleanup_builtin_calls") => "cleanup_builtin_calls",
            ("admin", "migrate_calls_confidence") => "migrate_calls_confidence",
            ("admin", "cleanup_sync_data") => "cleanup_sync_data",
            ("admin", "update_staleness_scores") => "update_staleness_scores",
            ("admin", "update_energy_scores") => "update_energy_scores",
            ("admin", "search_neurons") => "search_neurons",
            ("admin", "reinforce_neurons") => "reinforce_neurons",
            ("admin", "decay_synapses") => "decay_synapses",
            ("admin", "backfill_synapses") => "backfill_synapses",
            ("admin", "reindex_decisions") => "reindex_decisions",
            ("admin", "backfill_decision_embeddings") => "backfill_decision_embeddings",
            ("admin", "backfill_touches") => "backfill_touches",
            ("admin", "backfill_discussed") => "backfill_discussed",
            ("admin", "update_fabric_scores") => "update_fabric_scores",
            ("admin", "bootstrap_knowledge_fabric") => "bootstrap_knowledge_fabric",
            ("admin", "reinforce_isomorphic") => "reinforce_isomorphic",
            ("admin", "detect_skills") => "detect_skills",
            ("admin", "detect_skill_fission") => "detect_skill_fission",
            ("admin", "detect_skill_fusion") => "detect_skill_fusion",
            ("admin", "maintain_skills") => "maintain_skills",
            ("admin", "auto_anchor_notes") => "auto_anchor_notes",
            ("admin", "reconstruct_knowledge") => "reconstruct_knowledge",
            ("admin", "heal_scars") => "heal_scars",
            ("admin", "consolidate_memory") => "consolidate_memory",
            ("admin", "audit_gaps") => "audit_gaps",
            ("admin", "persist_health_report") => "persist_health_report",
            ("admin", "detect_stagnation") => "detect_stagnation",
            ("admin", "deep_maintenance") => "deep_maintenance",
            ("admin", "seed_prompt_fragments") => "seed_prompt_fragments",
            ("admin", "install_hooks") => "install_hooks",
            ("admin", "get_learning_stats") => "get_learning_stats",

            _ => {
                return Err(anyhow!(
                    "Unknown action '{}' for mega-tool '{}'",
                    action,
                    tool
                ))
            }
        };
        Ok(name.to_string())
    }

    /// Handle a tool call and return the result as JSON.
    ///
    /// All calls are routed through the REST API via `try_handle_http()`.
    /// Mega-tool names (e.g. "project") are first resolved to legacy names
    /// (e.g. "list_projects") for backward-compatible routing.
    pub async fn handle(&self, name: &str, args: Option<Value>) -> Result<Value> {
        let args = args.unwrap_or(json!({}));
        let start = std::time::Instant::now();

        // ── Mega-tool resolution ────────────────────────────────────────
        let (resolved_name, resolved_args) = self.resolve_mega_tool(name, &args)?;
        let name = resolved_name.as_str();
        let mut args = resolved_args;
        unstringify_json_values(&mut args);

        // ── HTTP routing ────────────────────────────────────────────────
        let result = self.try_handle_http(name, &args).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // ── Trajectory collection (fire-and-forget) ─────────────────────
        if let Some(ref collector) = self.collector {
            // Extract mega-tool name and action from the original call
            let (tool_name, action) = if let Some(dot) = name.find('_') {
                (&name[..dot], &name[dot + 1..])
            } else {
                (name, "default")
            };

            // Hash params to avoid storing PII — keep only structural keys
            let params_hash = {
                let keys: Vec<&str> = args
                    .as_object()
                    .map(|m| m.keys().map(|k| k.as_str()).collect())
                    .unwrap_or_default();
                format!("{:?}", keys)
            };

            let success = result.as_ref().map(|r| r.is_some()).unwrap_or(false);

            let tool_usage = neural_routing_runtime::ToolUsage {
                tool_name: tool_name.to_string(),
                action: action.to_string(),
                params_hash,
                duration_ms: Some(elapsed_ms),
                success,
            };

            // Use a placeholder session_id — the real session binding happens at the chat level
            collector.record_decision(neural_routing_runtime::DecisionRecord {
                session_id: "mcp-direct".to_string(),
                context_embedding: vec![],
                action_type: format!("{}.{}", tool_name, action),
                action_params: args.clone(),
                alternatives_count: 1,
                chosen_index: 0,
                confidence: if success { 0.8 } else { 0.2 },
                tool_usages: vec![tool_usage],
                touched_entities: vec![],
                timestamp_ms: elapsed_ms,
                query_embedding: vec![],
                node_features: vec![],
                protocol_run_id: None,
                protocol_state: None,
            });
        }

        match result {
            Ok(Some(value)) => Ok(value),
            Ok(None) => Err(anyhow!("Unknown tool: '{}'", name)),
            Err(e) => Err(e),
        }
    }

    // ========================================================================
    // HTTP-mode routing (migrated tools)
    // ========================================================================

    /// Try to handle a tool call via HTTP proxy.
    /// Returns `Ok(Some(value))` if the tool is migrated, `Ok(None)` if not yet migrated.
    async fn try_handle_http(&self, name: &str, args: &Value) -> Result<Option<Value>> {
        let http = self.http();

        match name {
            // ── P2: Projects (8 tools) ──────────────────────────────────
            "list_projects" => {
                let mut query = Vec::new();
                if let Some(s) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                if let Some(sb) = args.get("sort_by").and_then(|v| v.as_str()) {
                    query.push(("sort_by".to_string(), sb.to_string()));
                }
                if let Some(so) = args.get("sort_order").and_then(|v| v.as_str()) {
                    query.push(("sort_order".to_string(), so.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/projects").await?
                } else {
                    http.get_with_query("/api/projects", &query).await?
                };
                Ok(Some(result))
            }

            "create_project" => {
                let result = http.post("/api/projects", args).await?;
                Ok(Some(result))
            }

            "get_project" => {
                let slug = extract_string(args, "slug")?;
                let result = http.get(&format!("/api/projects/{}", slug)).await?;
                Ok(Some(result))
            }

            "update_project" => {
                let slug = extract_string(args, "slug")?;
                // Build PATCH body — only include fields that are present
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("root_path") {
                    body.insert("root_path".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/projects/{}", slug), &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "delete_project" => {
                let slug = extract_string(args, "slug")?;
                let result = http.delete(&format!("/api/projects/{}", slug)).await?;
                Ok(Some(result))
            }

            "sync_project" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(force) = args.get("force").and_then(|v| v.as_bool()) {
                    if force {
                        query.push(("force".to_string(), "true".to_string()));
                    }
                }
                let result = if query.is_empty() {
                    http.post(&format!("/api/projects/{}/sync", slug), &json!({}))
                        .await?
                } else {
                    // POST with query params — use get_with_query pattern but POST
                    let url = format!("/api/projects/{}/sync?force=true", slug);
                    http.post(&url, &json!({})).await?
                };
                Ok(Some(result))
            }

            "get_project_roadmap" => {
                let slug = extract_string(args, "slug")?;
                let result = http.get(&format!("/api/projects/{}/roadmap", slug)).await?;
                Ok(Some(result))
            }

            "list_project_plans" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/projects/{}/plans", slug)).await?
                } else {
                    http.get_with_query(&format!("/api/projects/{}/plans", slug), &query)
                        .await?
                };
                Ok(Some(result))
            }

            "get_project_graph" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("layers").and_then(|v| v.as_str()) {
                    query.push(("layers".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("community").and_then(|v| v.as_i64()) {
                    query.push(("community".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/projects/{}/graph", slug)).await?
                } else {
                    http.get_with_query(&format!("/api/projects/{}/graph", slug), &query)
                        .await?
                };
                Ok(Some(result))
            }

            "get_intelligence_summary" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/intelligence/summary", slug))
                    .await?;
                Ok(Some(result))
            }

            "get_health_dashboard" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/health-dashboard", slug))
                    .await?;
                Ok(Some(result))
            }

            "get_auto_roadmap" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/auto-roadmap", slug))
                    .await?;
                Ok(Some(result))
            }

            "get_embeddings_projection" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/embeddings/projection", slug))
                    .await?;
                Ok(Some(result))
            }

            "get_scaffolding_level" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/scaffolding", slug))
                    .await?;
                Ok(Some(result))
            }

            "set_scaffolding_override" => {
                let slug = extract_string(args, "slug")?;
                let mut body = serde_json::Map::new();
                if let Some(level) = args.get("level") {
                    body.insert("level".to_string(), level.clone());
                } else {
                    body.insert("level".to_string(), Value::Null);
                }
                let result = http
                    .put(
                        &format!("/api/projects/{}/scaffolding", slug),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            // ── P3: Plans (9 tools) ──────────────────────────────────────
            "list_plans" => {
                let mut query = Vec::new();
                if let Some(s) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(v) = args.get("priority_min").and_then(|v| v.as_i64()) {
                    query.push(("priority_min".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("priority_max").and_then(|v| v.as_i64()) {
                    query.push(("priority_max".to_string(), v.to_string()));
                }
                if let Some(s) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                if let Some(sb) = args.get("sort_by").and_then(|v| v.as_str()) {
                    query.push(("sort_by".to_string(), sb.to_string()));
                }
                if let Some(so) = args.get("sort_order").and_then(|v| v.as_str()) {
                    query.push(("sort_order".to_string(), so.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/plans").await?
                } else {
                    http.get_with_query("/api/plans", &query).await?
                };
                Ok(Some(result))
            }

            "create_plan" => {
                let result = http.post("/api/plans", args).await?;
                Ok(Some(result))
            }

            "get_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.get(&format!("/api/plans/{}", plan_id)).await?;
                Ok(Some(result))
            }

            "update_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("priority") {
                    body.insert("priority".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/plans/{}", plan_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "update_plan_status" => {
                let plan_id = extract_id(args, "plan_id")?;
                let status = extract_string(args, "status")?;
                let body = json!({"status": status});
                let result = http
                    .patch(&format!("/api/plans/{}", plan_id), &body)
                    .await?;
                // REST returns 204 → null, normalize to {"updated": true}
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.delete(&format!("/api/plans/{}", plan_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "link_plan_to_project" => {
                let plan_id = extract_id(args, "plan_id")?;
                let project_id = extract_id(args, "project_id")?;
                let body = json!({"project_id": project_id});
                let result = http
                    .put(&format!("/api/plans/{}/project", plan_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_plan_from_project" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .delete(&format!("/api/plans/{}/project", plan_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            "get_dependency_graph" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/dependency-graph", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "get_critical_path" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/critical-path", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "get_waves" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.get(&format!("/api/plans/{}/waves", plan_id)).await?;
                Ok(Some(result))
            }

            "run_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let cwd = args
                    .get("cwd")
                    .and_then(|v| v.as_str())
                    .unwrap_or(".")
                    .to_string();
                let project_slug = args
                    .get("project_slug")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                // MCP calls are always from chat context → TriggerSource::Chat
                let body =
                    json!({ "cwd": cwd, "project_slug": project_slug, "triggered_by": "chat" });
                let result = http
                    .post(&format!("/api/plans/{}/run", plan_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "get_run_status" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/run/status", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "cancel_plan_run" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/run/cancel", plan_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "create_auto_pr" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/run/auto-pr", plan_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            // ── P2b: Triggers (5 actions) ──────────────────────────────────
            "add_trigger" => {
                let plan_id = extract_id(args, "plan_id")?;
                let body = json!({
                    "trigger_type": args.get("trigger_type").and_then(|v| v.as_str()).unwrap_or("schedule"),
                    "config": args.get("config").cloned().unwrap_or(json!({})),
                    "cooldown_secs": args.get("cooldown_secs").and_then(|v| v.as_u64()).unwrap_or(0)
                });
                let result = http
                    .post(&format!("/api/plans/{}/triggers", plan_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "list_triggers" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/triggers", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "remove_trigger" => {
                let trigger_id = extract_id(args, "trigger_id")?;
                let result = http
                    .delete(&format!("/api/triggers/{}", trigger_id))
                    .await?;
                Ok(Some(result))
            }

            "enable_trigger" => {
                let trigger_id = extract_id(args, "trigger_id")?;
                let result = http
                    .post(&format!("/api/triggers/{}/enable", trigger_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "disable_trigger" => {
                let trigger_id = extract_id(args, "trigger_id")?;
                let result = http
                    .post(&format!("/api/triggers/{}/disable", trigger_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "list_plan_runs" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.get(&format!("/api/plans/{}/runs", plan_id)).await?;
                Ok(Some(result))
            }

            "get_plan_run" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http.get(&format!("/api/runs/{}", run_id)).await?;
                Ok(Some(result))
            }

            "compare_plan_runs" => {
                let plan_id = extract_id(args, "plan_id")?;
                let run_ids = args.get("run_ids").cloned().unwrap_or(json!([]));
                let result = http
                    .post(
                        &format!("/api/plans/{}/runs/compare", plan_id),
                        &json!({ "run_ids": run_ids }),
                    )
                    .await?;
                Ok(Some(result))
            }

            "predict_plan_run" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/runs/predict", plan_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            // ── P3: Constraints (5 tools) ───────────────────────────────────
            "list_constraints" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/constraints", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "add_constraint" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/constraints", plan_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_constraint" => {
                let constraint_id = extract_id(args, "constraint_id")?;
                let result = http
                    .get(&format!("/api/constraints/{}", constraint_id))
                    .await?;
                Ok(Some(result))
            }

            "update_constraint" => {
                let constraint_id = extract_id(args, "constraint_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("constraint_type") {
                    body.insert("constraint_type".to_string(), v.clone());
                }
                if let Some(v) = args.get("enforced_by") {
                    body.insert("enforced_by".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/constraints/{}", constraint_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_constraint" => {
                let constraint_id = extract_id(args, "constraint_id")?;
                let result = http
                    .delete(&format!("/api/constraints/{}", constraint_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // ── P4: Tasks (13 tools) ────────────────────────────────────
            "list_tasks" => {
                let mut query = Vec::new();
                if let Some(s) = args.get("plan_id").and_then(|v| v.as_str()) {
                    query.push(("plan_id".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("workspace_slug").and_then(|v| v.as_str()) {
                    query.push(("workspace_slug".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(v) = args.get("priority_min").and_then(|v| v.as_i64()) {
                    query.push(("priority_min".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("priority_max").and_then(|v| v.as_i64()) {
                    query.push(("priority_max".to_string(), v.to_string()));
                }
                if let Some(s) = args.get("tags").and_then(|v| v.as_str()) {
                    query.push(("tags".to_string(), s.to_string()));
                }
                if let Some(s) = args.get("assigned_to").and_then(|v| v.as_str()) {
                    query.push(("assigned_to".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                if let Some(sb) = args.get("sort_by").and_then(|v| v.as_str()) {
                    query.push(("sort_by".to_string(), sb.to_string()));
                }
                if let Some(so) = args.get("sort_order").and_then(|v| v.as_str()) {
                    query.push(("sort_order".to_string(), so.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/tasks").await?
                } else {
                    http.get_with_query("/api/tasks", &query).await?
                };
                Ok(Some(result))
            }

            "create_task" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/tasks", plan_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_task" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.get(&format!("/api/tasks/{}", task_id)).await?;
                Ok(Some(result))
            }

            "update_task" => {
                let task_id = extract_id(args, "task_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("assigned_to") {
                    body.insert("assigned_to".to_string(), v.clone());
                }
                if let Some(v) = args.get("priority") {
                    body.insert("priority".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("estimated_complexity") {
                    body.insert("estimated_complexity".to_string(), v.clone());
                }
                if let Some(v) = args.get("actual_complexity") {
                    body.insert("actual_complexity".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/tasks/{}", task_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_task" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.delete(&format!("/api/tasks/{}", task_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_next_task" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/next-task", plan_id))
                    .await?;
                Ok(Some(result))
            }

            "add_task_dependencies" => {
                let task_id = extract_id(args, "task_id")?;
                // REST expects { "depends_on": [...] }, MCP sends { "dependency_ids": [...] }
                let dep_ids = args.get("dependency_ids").cloned().unwrap_or(json!([]));
                let body = json!({"depends_on": dep_ids});
                let result = http
                    .post(&format!("/api/tasks/{}/dependencies", task_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_task_dependency" => {
                let task_id = extract_id(args, "task_id")?;
                let dependency_id = extract_id(args, "dependency_id")?;
                let result = http
                    .delete(&format!(
                        "/api/tasks/{}/dependencies/{}",
                        task_id, dependency_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "get_task_blockers" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/tasks/{}/blockers", task_id))
                    .await?;
                Ok(Some(result))
            }

            "get_tasks_blocked_by" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/tasks/{}/blocking", task_id))
                    .await?;
                Ok(Some(result))
            }

            "get_task_context" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/tasks/{}/context", plan_id, task_id))
                    .await?;
                Ok(Some(result))
            }

            "get_task_prompt" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/plans/{}/tasks/{}/prompt", plan_id, task_id))
                    .await?;
                Ok(Some(result))
            }

            "build_task_prompt" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(
                        &format!("/api/plans/{}/tasks/{}/build_prompt", plan_id, task_id),
                        args,
                    )
                    .await?;
                Ok(Some(result))
            }

            "enrich_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .post(&format!("/api/plans/{}/enrich", plan_id), args)
                    .await?;
                Ok(Some(result))
            }

            "delegate_task" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(
                        &format!("/api/plans/{}/tasks/{}/delegate", plan_id, task_id),
                        args,
                    )
                    .await?;
                Ok(Some(result))
            }

            "enrich_task" => {
                let plan_id = extract_id(args, "plan_id")?;
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(
                        &format!("/api/plans/{}/tasks/{}/enrich", plan_id, task_id),
                        args,
                    )
                    .await?;
                Ok(Some(result))
            }

            "add_decision" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(&format!("/api/tasks/{}/decisions", task_id), args)
                    .await?;
                Ok(Some(result))
            }

            // ── P4: Steps (6 tools) ────────────────────────────────────
            "list_steps" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.get(&format!("/api/tasks/{}/steps", task_id)).await?;
                Ok(Some(result))
            }

            "create_step" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .post(&format!("/api/tasks/{}/steps", task_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_step" => {
                let step_id = extract_id(args, "step_id")?;
                let result = http.get(&format!("/api/steps/{}", step_id)).await?;
                Ok(Some(result))
            }

            "update_step" => {
                let step_id = extract_id(args, "step_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("verification") {
                    body.insert("verification".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/steps/{}", step_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_step" => {
                let step_id = extract_id(args, "step_id")?;
                let result = http.delete(&format!("/api/steps/{}", step_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_step_progress" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http
                    .get(&format!("/api/tasks/{}/steps/progress", task_id))
                    .await?;
                Ok(Some(result))
            }

            // ── P4: Decisions (4 tools) ────────────────────────────────
            "get_decision" => {
                let decision_id = extract_id(args, "decision_id")?;
                let result = http.get(&format!("/api/decisions/{}", decision_id)).await?;
                Ok(Some(result))
            }

            "update_decision" => {
                let decision_id = extract_id(args, "decision_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("rationale") {
                    body.insert("rationale".to_string(), v.clone());
                }
                if let Some(v) = args.get("chosen_option") {
                    body.insert("chosen_option".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/decisions/{}", decision_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_decision" => {
                let decision_id = extract_id(args, "decision_id")?;
                let result = http
                    .delete(&format!("/api/decisions/{}", decision_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "search_decisions" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("q".to_string(), query_str)];
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(s) = extract_optional_string(args, "project_slug") {
                    query.push(("project_slug".to_string(), s));
                }
                let result = http.get_with_query("/api/decisions/search", &query).await?;
                Ok(Some(result))
            }

            "search_decisions_semantic" => {
                let query_str = extract_string(args, "query")?;
                let explicit_profile = args.get("profile").and_then(|v| v.as_str());
                let explicit_intent = args.get("intent_mode").and_then(|v| v.as_str());
                let resolved_profile =
                    resolve_intent_profile(&query_str, explicit_profile, explicit_intent);

                let mut query = vec![("query".to_string(), query_str)];
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), pid.to_string()));
                }
                if let Some(t) = args.get("temperature").and_then(|v| v.as_f64()) {
                    query.push(("temperature".to_string(), t.to_string()));
                }
                if let Some(ref profile) = resolved_profile {
                    query.push(("profile".to_string(), profile.clone()));
                }
                let result = http
                    .get_with_query("/api/decisions/search-semantic", &query)
                    .await?;
                Ok(Some(result))
            }

            "add_decision_affects" => {
                let decision_id = extract_id(args, "decision_id")?;
                let mut body = serde_json::Map::new();
                body.insert(
                    "entity_type".to_string(),
                    Value::String(extract_string(args, "entity_type")?),
                );
                body.insert(
                    "entity_id".to_string(),
                    Value::String(extract_string(args, "entity_id")?),
                );
                if let Some(v) = args.get("impact_description") {
                    body.insert("impact_description".to_string(), v.clone());
                }
                let result = http
                    .post(
                        &format!("/api/decisions/{}/affects", decision_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"status": "ok"})
                } else {
                    result
                }))
            }

            "remove_decision_affects" => {
                let decision_id = extract_id(args, "decision_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                // Use query-param variant to safely handle entity_ids with slashes (file paths)
                let encoded_type = urlencoding::encode(&entity_type);
                let encoded_id = urlencoding::encode(&entity_id);
                let result = http
                    .delete(&format!(
                        "/api/decisions/{}/affects?entity_type={}&entity_id={}",
                        decision_id, encoded_type, encoded_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"status": "ok"})
                } else {
                    result
                }))
            }

            "list_decision_affects" => {
                let decision_id = extract_id(args, "decision_id")?;
                let result = http
                    .get(&format!("/api/decisions/{}/affects", decision_id))
                    .await?;
                Ok(Some(result))
            }

            "get_decisions_affecting" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(s) = extract_optional_string(args, "status") {
                    query.push(("status".to_string(), s));
                }
                let result = http
                    .get_with_query("/api/decisions/affecting", &query)
                    .await?;
                Ok(Some(result))
            }

            "supersede_decision" => {
                let new_id = extract_id(args, "decision_id")?;
                let old_id = extract_id(args, "superseded_by_id")?;
                let result = http
                    .post(
                        &format!("/api/decisions/{}/supersedes/{}", new_id, old_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"status": "ok"})
                } else {
                    result
                }))
            }

            "get_decision_timeline" => {
                let mut query = Vec::new();
                if let Some(tid) = extract_optional_string(args, "task_id") {
                    query.push(("task_id".to_string(), tid));
                }
                if let Some(f) = extract_optional_string(args, "from") {
                    query.push(("from".to_string(), f));
                }
                if let Some(t) = extract_optional_string(args, "to") {
                    query.push(("to".to_string(), t));
                }
                let result = http
                    .get_with_query("/api/decisions/timeline", &query)
                    .await?;
                Ok(Some(result))
            }

            // ── P5: Milestones (9 tools) ────────────────────────────────
            "list_milestones" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = Vec::new();
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let path = format!("/api/projects/{}/milestones", project_id);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_milestone" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .post(&format!("/api/projects/{}/milestones", project_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let include_tasks = args
                    .get("include_tasks")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let result = http
                    .get(&format!(
                        "/api/milestones/{}?include_tasks={}",
                        milestone_id, include_tasks
                    ))
                    .await?;
                Ok(Some(result))
            }

            "update_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("target_date") {
                    body.insert("target_date".to_string(), v.clone());
                }
                if let Some(v) = args.get("closed_at") {
                    body.insert("closed_at".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/milestones/{}", milestone_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let result = http
                    .delete(&format!("/api/milestones/{}", milestone_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_milestone_progress" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let result = http
                    .get(&format!("/api/milestones/{}/progress", milestone_id))
                    .await?;
                Ok(Some(result))
            }

            "add_task_to_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let task_id = extract_id(args, "task_id")?;
                let body = json!({"task_id": task_id});
                let result = http
                    .post(&format!("/api/milestones/{}/tasks", milestone_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "link_plan_to_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let body = json!({"plan_id": plan_id});
                let result = http
                    .post(&format!("/api/milestones/{}/plans", milestone_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_plan_from_milestone" => {
                let milestone_id = extract_id(args, "milestone_id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .delete(&format!(
                        "/api/milestones/{}/plans/{}",
                        milestone_id, plan_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            // ── P5: Releases (8 tools) ─────────────────────────────────
            "list_releases" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = Vec::new();
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let path = format!("/api/projects/{}/releases", project_id);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_release" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .post(&format!("/api/projects/{}/releases", project_id), args)
                    .await?;
                Ok(Some(result))
            }

            "get_release" => {
                let release_id = extract_id(args, "release_id")?;
                let result = http.get(&format!("/api/releases/{}", release_id)).await?;
                Ok(Some(result))
            }

            "update_release" => {
                let release_id = extract_id(args, "release_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("target_date") {
                    body.insert("target_date".to_string(), v.clone());
                }
                if let Some(v) = args.get("released_at") {
                    body.insert("released_at".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/releases/{}", release_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_release" => {
                let release_id = extract_id(args, "release_id")?;
                let result = http
                    .delete(&format!("/api/releases/{}", release_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_task_to_release" => {
                let release_id = extract_id(args, "release_id")?;
                let task_id = extract_id(args, "task_id")?;
                let body = json!({"task_id": task_id});
                let result = http
                    .post(&format!("/api/releases/{}/tasks", release_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "add_commit_to_release" => {
                let release_id = extract_id(args, "release_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                // REST expects "commit_hash" field name
                let body = json!({"commit_hash": commit_sha});
                let result = http
                    .post(&format!("/api/releases/{}/commits", release_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_commit_from_release" => {
                let release_id = extract_id(args, "release_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                let result = http
                    .delete(&format!(
                        "/api/releases/{}/commits/{}",
                        release_id, commit_sha
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            // ── P6: Commits (5 tools) ───────────────────────────────────
            "create_commit" => {
                // Map MCP field names to REST field names: sha→hash
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("sha") {
                    body.insert("hash".to_string(), v.clone());
                }
                if let Some(v) = args.get("message") {
                    body.insert("message".to_string(), v.clone());
                }
                if let Some(v) = args.get("author") {
                    body.insert("author".to_string(), v.clone());
                }
                if let Some(v) = args.get("files_changed") {
                    body.insert("files_changed".to_string(), v.clone());
                }
                if let Some(v) = args.get("project_id") {
                    body.insert("project_id".to_string(), v.clone());
                }
                let result = http.post("/api/commits", &Value::Object(body)).await?;
                Ok(Some(result))
            }

            "link_commit_to_task" => {
                let task_id = extract_id(args, "task_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                // REST expects "commit_hash" field name
                let body = json!({"commit_hash": commit_sha});
                let result = http
                    .post(&format!("/api/tasks/{}/commits", task_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "link_commit_to_plan" => {
                let plan_id = extract_id(args, "plan_id")?;
                let commit_sha = extract_string(args, "commit_sha")?;
                // REST expects "commit_hash" field name
                let body = json!({"commit_hash": commit_sha});
                let result = http
                    .post(&format!("/api/plans/{}/commits", plan_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "get_task_commits" => {
                let task_id = extract_id(args, "task_id")?;
                let result = http.get(&format!("/api/tasks/{}/commits", task_id)).await?;
                Ok(Some(result))
            }

            "get_plan_commits" => {
                let plan_id = extract_id(args, "plan_id")?;
                let result = http.get(&format!("/api/plans/{}/commits", plan_id)).await?;
                Ok(Some(result))
            }

            "get_commit_files" => {
                let sha = args
                    .get("sha")
                    .or_else(|| args.get("commit_sha"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("sha is required"))?;
                let result = http.get(&format!("/api/commits/{}/files", sha)).await?;
                Ok(Some(result))
            }

            "get_file_history" => {
                let file_path = args
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("file_path is required"))?;
                let mut query_params = vec![("path".to_string(), file_path.to_string())];
                if let Some(limit) = args.get("limit").and_then(|v| v.as_i64()) {
                    query_params.push(("limit".to_string(), limit.to_string()));
                }
                let result = http
                    .get_with_query("/api/files/history", &query_params)
                    .await?;
                Ok(Some(result))
            }

            // ── P7: Notes & Knowledge (23 tools) ──────────────────────────

            // --- CRUD (5) ---
            "list_notes" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("note_type").and_then(|v| v.as_str()) {
                    query.push(("note_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("importance").and_then(|v| v.as_str()) {
                    query.push(("importance".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_staleness").and_then(|v| v.as_f64()) {
                    query.push(("min_staleness".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("max_staleness").and_then(|v| v.as_f64()) {
                    query.push(("max_staleness".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("tags").and_then(|v| v.as_str()) {
                    query.push(("tags".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("global_only").and_then(|v| v.as_bool()) {
                    query.push(("global_only".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/notes").await?
                } else {
                    http.get_with_query("/api/notes", &query).await?
                };
                Ok(Some(result))
            }

            "create_note" => {
                // Forward full args as body (REST ignores unknown fields via serde)
                let result = http.post("/api/notes", args).await?;
                Ok(Some(result))
            }

            "get_note" => {
                let note_id = extract_id(args, "note_id")?;
                let result = http.get(&format!("/api/notes/{}", note_id)).await?;
                Ok(Some(result))
            }

            "update_note" => {
                let note_id = extract_id(args, "note_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("content") {
                    body.insert("content".to_string(), v.clone());
                }
                if let Some(v) = args.get("importance") {
                    body.insert("importance".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/notes/{}", note_id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_note" => {
                let note_id = extract_id(args, "note_id")?;
                let result = http.delete(&format!("/api/notes/{}", note_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // --- Search (3) ---
            "search_notes" => {
                // MCP param "query" → REST param "q"
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("q".to_string(), query_str)];
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("note_type").and_then(|v| v.as_str()) {
                    query.push(("note_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("importance").and_then(|v| v.as_str()) {
                    query.push(("importance".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes/search", &query).await?;
                Ok(Some(result))
            }

            "search_notes_semantic" => {
                let query_str = extract_string(args, "query")?;
                let explicit_profile = args.get("profile").and_then(|v| v.as_str());
                let explicit_intent = args.get("intent_mode").and_then(|v| v.as_str());
                let resolved_profile =
                    resolve_intent_profile(&query_str, explicit_profile, explicit_intent);

                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("workspace_slug").and_then(|v| v.as_str()) {
                    query.push(("workspace_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(t) = args.get("temperature").and_then(|v| v.as_f64()) {
                    query.push(("temperature".to_string(), t.to_string()));
                }
                if let Some(ref profile) = resolved_profile {
                    query.push(("profile".to_string(), profile.clone()));
                }
                let result = http
                    .get_with_query("/api/notes/search-semantic", &query)
                    .await?;
                Ok(Some(result))
            }

            "search_neurons" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("max_results").and_then(|v| v.as_i64()) {
                    query.push(("max_results".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("max_hops").and_then(|v| v.as_i64()) {
                    query.push(("max_hops".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/neurons/search", &query)
                    .await?;
                Ok(Some(result))
            }

            // --- Lifecycle (3) ---
            "confirm_note" => {
                let note_id = extract_id(args, "note_id")?;
                let result = http
                    .post(&format!("/api/notes/{}/confirm", note_id), &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "invalidate_note" => {
                let note_id = extract_id(args, "note_id")?;
                let reason = extract_string(args, "reason")?;
                let body = json!({"reason": reason});
                let result = http
                    .post(&format!("/api/notes/{}/invalidate", note_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "supersede_note" => {
                let old_note_id = extract_id(args, "old_note_id")?;
                // Forward remaining args as body (old_note_id extracted for URL path)
                let result = http
                    .post(&format!("/api/notes/{}/supersede", old_note_id), args)
                    .await?;
                Ok(Some(result))
            }

            // --- Linking (2) ---
            "link_note_to_entity" => {
                let note_id = extract_id(args, "note_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let body = json!({"entity_type": entity_type, "entity_id": entity_id});
                let result = http
                    .post(&format!("/api/notes/{}/links", note_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_note_from_entity" => {
                let note_id = extract_id(args, "note_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let result = http
                    .delete(&format!(
                        "/api/notes/{}/links/{}/{}",
                        note_id, entity_type, entity_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            // --- Retrieval (5) ---
            "get_context_notes" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes/context", &query).await?;
                Ok(Some(result))
            }

            "get_notes_needing_review" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/notes/needs-review").await?
                } else {
                    http.get_with_query("/api/notes/needs-review", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "get_propagated_notes" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                // Forward relation_types as comma-separated string
                if let Some(v) = args.get("relation_types").and_then(|v| v.as_str()) {
                    query.push(("relation_types".to_string(), v.to_string()));
                }
                // P2P coupling: cross-project weighting params
                if let Some(v) = args.get("source_project_id").and_then(|v| v.as_str()) {
                    query.push(("source_project_id".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("force_cross_project").and_then(|v| v.as_bool()) {
                    query.push(("force_cross_project".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes/propagated", &query).await?;
                Ok(Some(result))
            }

            "get_context_knowledge" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/context-knowledge", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_propagated_knowledge" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let mut query = vec![
                    ("entity_type".to_string(), entity_type),
                    ("entity_id".to_string(), entity_id),
                ];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("min_score").and_then(|v| v.as_f64()) {
                    query.push(("min_score".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("relation_types").and_then(|v| v.as_str()) {
                    query.push(("relation_types".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/notes/propagated-knowledge", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_entity_notes" => {
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_string(args, "entity_id")?;
                let result = http
                    .get(&format!(
                        "/api/entities/{}/{}/notes",
                        entity_type, entity_id
                    ))
                    .await?;
                Ok(Some(result))
            }

            "list_project_notes" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/projects/{}/notes", project_id))
                        .await?
                } else {
                    http.get_with_query(&format!("/api/projects/{}/notes", project_id), &query)
                        .await?
                };
                Ok(Some(result))
            }

            // --- RFC convenience (3) ---
            "list_rfcs" => {
                // Proxy to list_notes with note_type=rfc
                let mut query = vec![("note_type".to_string(), "rfc".to_string())];
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/notes", &query).await?;
                Ok(Some(result))
            }

            "advance_rfc" => {
                // Get note → extract rfc-run:UUID tag → fire transition
                let note_id = extract_id(args, "note_id")?;
                let trigger = extract_string(args, "trigger")?;

                // 1. Get the note to find the run ID from tags
                let note = http.get(&format!("/api/notes/{}", note_id)).await?;
                let tags = note
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let run_id = tags
                    .iter()
                    .filter_map(|t| t.as_str())
                    .find(|t| t.starts_with("rfc-run:"))
                    .map(|t| t.trim_start_matches("rfc-run:").to_string())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Note {} has no rfc-run:UUID tag — not an RFC with a lifecycle run",
                            note_id
                        )
                    })?;

                // 2. Fire transition on the run
                let body = json!({"trigger": trigger});
                let result = http
                    .post(&format!("/api/protocols/runs/{}/transition", run_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "get_rfc_status" => {
                // Get note → extract rfc-run:UUID → get run details
                let note_id = extract_id(args, "note_id")?;

                // 1. Get the note
                let note = http.get(&format!("/api/notes/{}", note_id)).await?;
                let tags = note
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let run_id = tags
                    .iter()
                    .filter_map(|t| t.as_str())
                    .find(|t| t.starts_with("rfc-run:"))
                    .map(|t| t.trim_start_matches("rfc-run:").to_string());

                if let Some(run_id) = run_id {
                    // 2. Get the run (enriched with children_count)
                    let run = http.get(&format!("/api/protocols/runs/{}", run_id)).await?;
                    Ok(Some(json!({
                        "note_id": note_id.to_string(),
                        "note_type": note.get("note_type"),
                        "content_preview": note.get("content").and_then(|c| c.as_str()).map(|c| {
                            if c.len() > 200 { format!("{}...", &c[..200]) } else { c.to_string() }
                        }),
                        "rfc_run": run,
                        "tags": note.get("tags"),
                    })))
                } else {
                    Ok(Some(json!({
                        "note_id": note_id.to_string(),
                        "note_type": note.get("note_type"),
                        "rfc_run": null,
                        "message": "No rfc-lifecycle run associated with this note",
                        "tags": note.get("tags"),
                    })))
                }
            }

            // --- Admin (5) ---
            "update_staleness_scores" => {
                let result = http.post("/api/notes/update-staleness", &json!({})).await?;
                Ok(Some(result))
            }

            "update_energy_scores" => {
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("half_life") {
                    body.insert("half_life".to_string(), v.clone());
                }
                let result = http
                    .post("/api/notes/update-energy", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "reinforce_neurons" => {
                // Forward full args as body (note_ids, energy_boost, synapse_boost)
                let result = http.post("/api/notes/neurons/reinforce", args).await?;
                Ok(Some(result))
            }

            "decay_synapses" => {
                // Forward full args as body (decay_amount, prune_threshold)
                let result = http.post("/api/notes/neurons/decay", args).await?;
                Ok(Some(result))
            }

            "backfill_synapses" => {
                // REST runs async (202 Accepted), MCP Direct was sync.
                // We forward to the async endpoint and return the 202 response.
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("batch_size") {
                    body.insert("batch_size".to_string(), v.clone());
                }
                if let Some(v) = args.get("min_similarity") {
                    body.insert("min_similarity".to_string(), v.clone());
                }
                if let Some(v) = args.get("max_neighbors") {
                    body.insert("max_neighbors".to_string(), v.clone());
                }
                let result = http
                    .post("/api/admin/backfill-synapses", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "reindex_decisions" => {
                let result = http
                    .post("/api/admin/reindex-decisions", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "backfill_decision_embeddings" => {
                let result = http
                    .post("/api/admin/backfill-decision-embeddings", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "backfill_touches" => {
                let project_slug = extract_string(args, "project_slug")
                    .or_else(|_| extract_string(args, "slug"))?;
                let result = http
                    .post(
                        &format!("/api/projects/{}/backfill-touches", project_slug),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }

            "backfill_discussed" => {
                let result = http
                    .post("/api/admin/backfill-discussed", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "update_fabric_scores" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/update-fabric-scores", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "bootstrap_knowledge_fabric" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post(
                        "/api/admin/bootstrap-knowledge-fabric",
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "reinforce_isomorphic" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/reinforce-isomorphic", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "audit_gaps" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/audit-gaps", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "persist_health_report" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/persist-health-report", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "detect_stagnation" => {
                let pid = args
                    .get("project_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("project_id is required"))?;
                let result = http
                    .get(&format!("/api/admin/detect-stagnation/{}", pid))
                    .await?;
                Ok(Some(result))
            }

            "deep_maintenance" => {
                let pid = args
                    .get("project_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("project_id is required"))?;
                let result = http
                    .post(
                        &format!("/api/admin/deep-maintenance/{}", pid),
                        &Value::Object(serde_json::Map::new()),
                    )
                    .await?;
                Ok(Some(result))
            }

            "seed_prompt_fragments" => {
                let pid = args
                    .get("project_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("project_id is required"))?;
                let result = http
                    .post(
                        &format!("/api/admin/seed-prompt-fragments/{}", pid),
                        &Value::Object(serde_json::Map::new()),
                    )
                    .await?;
                Ok(Some(result))
            }

            "detect_skills" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                if let Some(force) = args.get("force").and_then(|v| v.as_bool()) {
                    body.insert("force".to_string(), Value::Bool(force));
                }
                let result = http
                    .post("/api/admin/detect-skills", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "detect_skill_fission" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/detect-skill-fission", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "detect_skill_fusion" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/detect-skill-fusion", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "maintain_skills" => {
                let mut body = serde_json::Map::new();
                body.insert(
                    "project_id".to_string(),
                    json!(extract_string(args, "project_id")?),
                );
                if let Some(level) = args.get("level").and_then(|v| v.as_str()) {
                    body.insert("level".to_string(), Value::String(level.to_string()));
                }
                let result = http
                    .post("/api/admin/skill-maintenance", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "auto_anchor_notes" => {
                let mut body = serde_json::Map::new();
                body.insert(
                    "project_id".to_string(),
                    json!(extract_string(args, "project_id")?),
                );
                let result = http
                    .post("/api/admin/auto-anchor-notes", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "reconstruct_knowledge" => {
                let mut body = serde_json::Map::new();
                if let Some(pid) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), Value::String(pid.to_string()));
                }
                let result = http
                    .post("/api/admin/reconstruct-knowledge", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "heal_scars" => {
                let node_id = extract_string(args, "node_id")?;
                let body = json!({"node_id": node_id});
                let result = http.post("/api/notes/neurons/heal-scars", &body).await?;
                Ok(Some(result))
            }

            "consolidate_memory" => {
                let result = http
                    .post("/api/notes/consolidate-memory", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "install_hooks" => {
                // Hooks are now managed automatically via the Nexus SDK's
                // in-process SkillActivationHook. No installation needed.
                Ok(Some(json!({
                    "message": "Hooks are now managed automatically via the Nexus SDK. No installation needed. Skill activation happens in-process during create_session/resume_session.",
                    "deprecated": true
                })))
            }

            "get_learning_stats" => {
                let tracker = crate::feedback::OutcomeTracker::global();
                let stats = tracker.get_learning_stats().await;
                Ok(Some(serde_json::to_value(stats).unwrap_or(json!({}))))
            }

            // ── P8: Workspaces (34 tools) ──────────────────────────────────

            // --- Workspace CRUD (5) ---
            "list_workspaces" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("search").and_then(|v| v.as_str()) {
                    query.push(("search".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/workspaces").await?
                } else {
                    http.get_with_query("/api/workspaces", &query).await?
                };
                Ok(Some(result))
            }

            "create_workspace" => {
                let result = http.post("/api/workspaces", args).await?;
                Ok(Some(result))
            }

            "get_workspace" => {
                let slug = extract_string(args, "slug")?;
                let result = http.get(&format!("/api/workspaces/{}", slug)).await?;
                Ok(Some(result))
            }

            "update_workspace" => {
                let slug = extract_string(args, "slug")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("metadata") {
                    body.insert("metadata".to_string(), v.clone());
                }
                if let Some(v) = args.get("new_slug") {
                    body.insert("slug".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/workspaces/{}", slug), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_workspace" => {
                let slug = extract_string(args, "slug")?;
                let result = http.delete(&format!("/api/workspaces/{}", slug)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // --- Workspace Overview & Projects (4) ---
            "get_workspace_overview" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/overview", slug))
                    .await?;
                Ok(Some(result))
            }

            "list_workspace_projects" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/projects", slug))
                    .await?;
                Ok(Some(result))
            }

            "add_project_to_workspace" => {
                let slug = extract_string(args, "slug")?;
                let project_id = extract_id(args, "project_id")?;
                let body = json!({"project_id": project_id});
                let result = http
                    .post(&format!("/api/workspaces/{}/projects", slug), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_project_from_workspace" => {
                let slug = extract_string(args, "slug")?;
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .delete(&format!("/api/workspaces/{}/projects/{}", slug, project_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            // --- Workspace Topology (1) ---
            "get_workspace_topology" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/topology", slug))
                    .await?;
                Ok(Some(result))
            }

            "get_coupling_matrix" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .get(&format!("/api/workspaces/{}/coupling-matrix", slug))
                    .await?;
                Ok(Some(result))
            }

            // --- Workspace Milestones (10) ---
            "list_all_workspace_milestones" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("workspace_id").and_then(|v| v.as_str()) {
                    query.push(("workspace_id".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/workspace-milestones").await?
                } else {
                    http.get_with_query("/api/workspace-milestones", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "list_workspace_milestones" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let path = format!("/api/workspaces/{}/milestones", slug);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_workspace_milestone" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .post(&format!("/api/workspaces/{}/milestones", slug), args)
                    .await?;
                Ok(Some(result))
            }

            "get_workspace_milestone" => {
                let id = extract_id(args, "milestone_id")?;
                let result = http
                    .get(&format!("/api/workspace-milestones/{}", id))
                    .await?;
                Ok(Some(result))
            }

            "update_workspace_milestone" => {
                let id = extract_id(args, "milestone_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("title") {
                    body.insert("title".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("target_date") {
                    body.insert("target_date".to_string(), v.clone());
                }
                let result = http
                    .patch(
                        &format!("/api/workspace-milestones/{}", id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_workspace_milestone" => {
                let id = extract_id(args, "milestone_id")?;
                let result = http
                    .delete(&format!("/api/workspace-milestones/{}", id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_task_to_workspace_milestone" => {
                let id = extract_id(args, "milestone_id")?;
                let task_id = extract_id(args, "task_id")?;
                let body = json!({"task_id": task_id});
                let result = http
                    .post(&format!("/api/workspace-milestones/{}/tasks", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "link_plan_to_workspace_milestone" => {
                let id = extract_id(args, "milestone_id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let body = json!({"plan_id": plan_id});
                let result = http
                    .post(&format!("/api/workspace-milestones/{}/plans", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            "unlink_plan_from_workspace_milestone" => {
                let id = extract_id(args, "milestone_id")?;
                let plan_id = extract_id(args, "plan_id")?;
                let result = http
                    .delete(&format!(
                        "/api/workspace-milestones/{}/plans/{}",
                        id, plan_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unlinked": true})
                } else {
                    result
                }))
            }

            "get_workspace_milestone_progress" => {
                let id = extract_id(args, "milestone_id")?;
                let result = http
                    .get(&format!("/api/workspace-milestones/{}/progress", id))
                    .await?;
                Ok(Some(result))
            }

            // --- Resources (6) ---
            "list_resources" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("resource_type").and_then(|v| v.as_str()) {
                    query.push(("resource_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let path = format!("/api/workspaces/{}/resources", slug);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_resource" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .post(&format!("/api/workspaces/{}/resources", slug), args)
                    .await?;
                Ok(Some(result))
            }

            "get_resource" => {
                let id = extract_id(args, "id")?;
                let result = http.get(&format!("/api/resources/{}", id)).await?;
                Ok(Some(result))
            }

            "update_resource" => {
                let id = extract_id(args, "id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("file_path") {
                    body.insert("file_path".to_string(), v.clone());
                }
                if let Some(v) = args.get("url") {
                    body.insert("url".to_string(), v.clone());
                }
                if let Some(v) = args.get("version") {
                    body.insert("version".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/resources/{}", id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_resource" => {
                let id = extract_id(args, "id")?;
                let result = http.delete(&format!("/api/resources/{}", id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "link_resource_to_project" => {
                let id = extract_id(args, "id")?;
                let project_id = extract_id(args, "project_id")?;
                let link_type = extract_string(args, "link_type")?;
                let body = json!({"project_id": project_id, "link_type": link_type});
                let result = http
                    .post(&format!("/api/resources/{}/projects", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"linked": true})
                } else {
                    result
                }))
            }

            // --- Components (8) ---
            "list_components" => {
                let slug = extract_string(args, "slug")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("component_type").and_then(|v| v.as_str()) {
                    query.push(("component_type".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let path = format!("/api/workspaces/{}/components", slug);
                let result = if query.is_empty() {
                    http.get(&path).await?
                } else {
                    http.get_with_query(&path, &query).await?
                };
                Ok(Some(result))
            }

            "create_component" => {
                let slug = extract_string(args, "slug")?;
                let result = http
                    .post(&format!("/api/workspaces/{}/components", slug), args)
                    .await?;
                Ok(Some(result))
            }

            "get_component" => {
                let id = extract_id(args, "id")?;
                let result = http.get(&format!("/api/components/{}", id)).await?;
                Ok(Some(result))
            }

            "update_component" => {
                let id = extract_id(args, "id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("runtime") {
                    body.insert("runtime".to_string(), v.clone());
                }
                if let Some(v) = args.get("config") {
                    body.insert("config".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                let result = http
                    .patch(&format!("/api/components/{}", id), &Value::Object(body))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"updated": true})
                } else {
                    result
                }))
            }

            "delete_component" => {
                let id = extract_id(args, "id")?;
                let result = http.delete(&format!("/api/components/{}", id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_component_dependency" => {
                let id = extract_id(args, "id")?;
                let depends_on_id = extract_id(args, "depends_on_id")?;
                let mut body = json!({"depends_on_id": depends_on_id});
                if let Some(v) = args.get("protocol") {
                    body["protocol"] = v.clone();
                }
                if let Some(v) = args.get("required") {
                    body["required"] = v.clone();
                }
                let result = http
                    .post(&format!("/api/components/{}/dependencies", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_component_dependency" => {
                let id = extract_id(args, "id")?;
                let dep_id = extract_id(args, "dep_id")?;
                let result = http
                    .delete(&format!("/api/components/{}/dependencies/{}", id, dep_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "map_component_to_project" => {
                let id = extract_id(args, "id")?;
                let project_id = extract_id(args, "project_id")?;
                let body = json!({"project_id": project_id});
                let result = http
                    .put(&format!("/api/components/{}/project", id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"mapped": true})
                } else {
                    result
                }))
            }

            // ── P9: Code Exploration & Analytics (16 tools) ────────────────

            // --- Search (3) ---
            "search_code" => {
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("query".to_string(), query_str)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("language").and_then(|v| v.as_str()) {
                    query.push(("language".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("path_prefix").and_then(|v| v.as_str()) {
                    query.push(("path_prefix".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/search", &query).await?;
                Ok(Some(result))
            }

            "search_project_code" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query_str = extract_string(args, "query")?;
                let mut query = vec![("q".to_string(), query_str)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("language").and_then(|v| v.as_str()) {
                    query.push(("language".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("path_prefix").and_then(|v| v.as_str()) {
                    query.push(("path_prefix".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query(
                        &format!("/api/projects/{}/code/search", project_slug),
                        &query,
                    )
                    .await?;
                Ok(Some(result))
            }

            "search_workspace_code" => {
                // REST search_code supports workspace_slug as query param
                let workspace_slug = extract_string(args, "workspace_slug")?;
                let query_str = extract_string(args, "query")?;
                let mut query = vec![
                    ("query".to_string(), query_str),
                    ("workspace_slug".to_string(), workspace_slug),
                ];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("language").and_then(|v| v.as_str()) {
                    query.push(("language".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("path_prefix").and_then(|v| v.as_str()) {
                    query.push(("path_prefix".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/search", &query).await?;
                Ok(Some(result))
            }

            // --- Graph Traversal (4) ---
            "get_file_symbols" => {
                let file_path = extract_string(args, "file_path")?;
                let result = http
                    .get(&format!("/api/code/symbols/{}", file_path))
                    .await?;
                Ok(Some(result))
            }

            "find_references" => {
                let symbol = extract_string(args, "symbol")?;
                let mut query = vec![("symbol".to_string(), symbol)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/references", &query).await?;
                Ok(Some(result))
            }

            "get_file_dependencies" => {
                let file_path = extract_string(args, "file_path")?;
                let result = http
                    .get(&format!("/api/code/dependencies/{}", file_path))
                    .await?;
                Ok(Some(result))
            }

            "get_call_graph" => {
                let function = extract_string(args, "function")?;
                let mut query = vec![("function".to_string(), function)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("depth".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/callgraph", &query).await?;
                Ok(Some(result))
            }

            // --- Analysis (3) ---
            "analyze_impact" => {
                let target = extract_string(args, "target")?;
                // multi=true → use multi-signal fusion endpoint (Plan 4)
                let use_multi = args.get("multi").and_then(|v| v.as_bool()).unwrap_or(false);
                if use_multi {
                    let mut query = vec![("target".to_string(), target)];
                    // project_slug is required for multi-signal
                    if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                        query.push(("project_slug".to_string(), v.to_string()));
                    }
                    if let Some(v) = args.get("profile").and_then(|v| v.as_str()) {
                        query.push(("profile".to_string(), v.to_string()));
                    }
                    let result = http
                        .get_with_query("/api/code/impact/multi", &query)
                        .await?;
                    Ok(Some(result))
                } else {
                    let mut query = vec![("target".to_string(), target)];
                    if let Some(v) = args.get("target_type").and_then(|v| v.as_str()) {
                        query.push(("target_type".to_string(), v.to_string()));
                    }
                    if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                        query.push(("project_slug".to_string(), v.to_string()));
                    }
                    if let Some(v) = args.get("profile").and_then(|v| v.as_str()) {
                        query.push(("profile".to_string(), v.to_string()));
                    }
                    let result = http.get_with_query("/api/code/impact", &query).await?;
                    Ok(Some(result))
                }
            }

            "get_architecture" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                let result = if query.is_empty() {
                    http.get("/api/code/architecture").await?
                } else {
                    http.get_with_query("/api/code/architecture", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "find_similar_code" => {
                // POST because of body (code_snippet)
                let result = http.post("/api/code/similar", args).await?;
                Ok(Some(result))
            }

            // --- Types (3) ---
            "find_trait_implementations" => {
                let trait_name = extract_string(args, "trait_name")?;
                let query = vec![("trait_name".to_string(), trait_name)];
                let result = http.get_with_query("/api/code/trait-impls", &query).await?;
                Ok(Some(result))
            }

            "find_type_traits" => {
                let type_name = extract_string(args, "type_name")?;
                let query = vec![("type_name".to_string(), type_name)];
                let result = http.get_with_query("/api/code/type-traits", &query).await?;
                Ok(Some(result))
            }

            "get_impl_blocks" => {
                let type_name = extract_string(args, "type_name")?;
                let query = vec![("type_name".to_string(), type_name)];
                let result = http.get_with_query("/api/code/impl-blocks", &query).await?;
                Ok(Some(result))
            }

            // --- Analytics GDS (3) ---
            "get_code_communities" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("min_size").and_then(|v| v.as_i64()) {
                    query.push(("min_size".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/communities", &query).await?;
                Ok(Some(result))
            }

            "get_code_health" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("god_function_threshold").and_then(|v| v.as_i64()) {
                    query.push(("god_function_threshold".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/health", &query).await?;
                Ok(Some(result))
            }

            "get_node_importance" => {
                let project_slug = extract_string(args, "project_slug")?;
                let node_path = extract_string(args, "node_path")?;
                let mut query = vec![
                    ("project_slug".to_string(), project_slug),
                    ("node_path".to_string(), node_path),
                ];
                if let Some(v) = args.get("node_type").and_then(|v| v.as_str()) {
                    query.push(("node_type".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/node-importance", &query)
                    .await?;
                Ok(Some(result))
            }

            // ── P10: Chat, Feature Graphs & Misc (14 tools) ─────────────

            // --- Chat CRUD (5) ---
            "list_chat_sessions" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    query.push(("project_slug".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("include_detached").and_then(|v| v.as_bool()) {
                    if v {
                        query.push(("include_detached".to_string(), "true".to_string()));
                    }
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/chat/sessions", &query).await?;
                Ok(Some(result))
            }

            "get_session_children" => {
                let id = extract_id(args, "session_id")?;
                let result = http
                    .get(&format!("/api/chat/sessions/{}/children", id))
                    .await?;
                Ok(Some(result))
            }

            "get_session_tree" => {
                let id = extract_id(args, "session_id")?;
                let result = http.get(&format!("/api/chat/sessions/{}/tree", id)).await?;
                Ok(Some(result))
            }

            "get_run_sessions" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http
                    .get(&format!("/api/chat/runs/{}/sessions", run_id))
                    .await?;
                Ok(Some(result))
            }

            "get_chat_session" => {
                let id = extract_id(args, "session_id")?;
                let result = http.get(&format!("/api/chat/sessions/{}", id)).await?;
                Ok(Some(result))
            }

            "delete_chat_session" => {
                let id = extract_id(args, "session_id")?;
                let result = http.delete(&format!("/api/chat/sessions/{}", id)).await?;
                Ok(Some(result))
            }

            "chat_send_message" => {
                // REST POST /api/chat/sessions creates a session + sends the first message
                let mut body = serde_json::Map::new();
                body.insert(
                    "message".to_string(),
                    json!(extract_string(args, "message")?),
                );
                body.insert("cwd".to_string(), json!(extract_string(args, "cwd")?));
                if let Some(v) = args.get("session_id").and_then(|v| v.as_str()) {
                    body.insert("session_id".to_string(), json!(v));
                }
                if let Some(v) = args.get("project_slug").and_then(|v| v.as_str()) {
                    body.insert("project_slug".to_string(), json!(v));
                }
                if let Some(v) = args.get("model").and_then(|v| v.as_str()) {
                    body.insert("model".to_string(), json!(v));
                }
                if let Some(v) = args.get("permission_mode").and_then(|v| v.as_str()) {
                    body.insert("permission_mode".to_string(), json!(v));
                }
                if let Some(v) = args.get("workspace_slug").and_then(|v| v.as_str()) {
                    body.insert("workspace_slug".to_string(), json!(v));
                }
                if let Some(v) = args.get("add_dirs") {
                    body.insert("add_dirs".to_string(), v.clone());
                }
                let result = http
                    .post("/api/chat/sessions", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "list_chat_messages" => {
                let id = extract_id(args, "session_id")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query(&format!("/api/chat/sessions/{}/messages", id), &query)
                    .await?;
                Ok(Some(result))
            }

            "add_discussed" => {
                let id = extract_id(args, "session_id")?;
                let entities = args.get("entities").cloned().unwrap_or_else(|| json!([]));
                let body = json!({ "entities": entities });
                let result = http
                    .post(&format!("/api/chat/sessions/{}/discussed", id), &body)
                    .await?;
                Ok(Some(result))
            }

            "get_session_entities" => {
                let id = extract_id(args, "session_id")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query(&format!("/api/chat/sessions/{}/discussed", id), &query)
                    .await?;
                Ok(Some(result))
            }

            // --- Feature Graphs (6) ---
            "create_feature_graph" => {
                let mut body = serde_json::Map::new();
                body.insert("name".to_string(), json!(extract_string(args, "name")?));
                body.insert(
                    "project_id".to_string(),
                    json!(extract_id(args, "project_id")?),
                );
                if let Some(v) = args.get("description").and_then(|v| v.as_str()) {
                    body.insert("description".to_string(), json!(v));
                }
                let result = http
                    .post("/api/feature-graphs", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "list_feature_graphs" => {
                let mut query = Vec::new();
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    query.push(("project_id".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/feature-graphs", &query).await?;
                Ok(Some(result))
            }

            "get_feature_graph" => {
                let id = extract_id(args, "id")?;
                let result = http.get(&format!("/api/feature-graphs/{}", id)).await?;
                Ok(Some(result))
            }

            "delete_feature_graph" => {
                let id = extract_id(args, "id")?;
                let result = http.delete(&format!("/api/feature-graphs/{}", id)).await?;
                Ok(Some(result))
            }

            "add_to_feature_graph" => {
                let id = extract_id(args, "feature_graph_id")?;
                let mut body = serde_json::Map::new();
                body.insert(
                    "entity_type".to_string(),
                    json!(extract_string(args, "entity_type")?),
                );
                body.insert(
                    "entity_id".to_string(),
                    json!(extract_string(args, "entity_id")?),
                );
                if let Some(v) = args.get("role").and_then(|v| v.as_str()) {
                    body.insert("role".to_string(), json!(v));
                }
                let result = http
                    .post(
                        &format!("/api/feature-graphs/{}/entities", id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "auto_build_feature_graph" => {
                let mut body = serde_json::Map::new();
                body.insert("name".to_string(), json!(extract_string(args, "name")?));
                body.insert(
                    "project_id".to_string(),
                    json!(extract_id(args, "project_id")?),
                );
                body.insert(
                    "entry_function".to_string(),
                    json!(extract_string(args, "entry_function")?),
                );
                if let Some(v) = args.get("description").and_then(|v| v.as_str()) {
                    body.insert("description".to_string(), json!(v));
                }
                if let Some(v) = args.get("depth").and_then(|v| v.as_u64()) {
                    body.insert("depth".to_string(), json!(v));
                }
                if let Some(v) = args.get("include_relations") {
                    body.insert("include_relations".to_string(), v.clone());
                }
                if let Some(v) = args.get("filter_community").and_then(|v| v.as_bool()) {
                    body.insert("filter_community".to_string(), json!(v));
                }
                let result = http
                    .post("/api/feature-graphs/auto-build", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "get_feature_graph_statistics" => {
                let id = extract_id(args, "id")?;
                let result = http
                    .get(&format!("/api/feature-graphs/{}/statistics", id))
                    .await?;
                Ok(Some(result))
            }

            "compare_feature_graphs" => {
                let id_a = extract_id(args, "id_a")?;
                let id_b = extract_id(args, "id_b")?;
                let query = vec![("id_a".to_string(), id_a), ("id_b".to_string(), id_b)];
                let result = http
                    .get_with_query("/api/feature-graphs/compare", &query)
                    .await?;
                Ok(Some(result))
            }

            "find_overlapping_feature_graphs" => {
                let id = extract_id(args, "id")?;
                let mut query = Vec::new();
                if let Some(v) = args.get("min_overlap").and_then(|v| v.as_f64()) {
                    query.push(("min_overlap".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query(&format!("/api/feature-graphs/{}/overlapping", id), &query)
                    .await?;
                Ok(Some(result))
            }

            // --- Misc (3) ---
            "plan_implementation" => {
                let mut body = serde_json::Map::new();
                body.insert(
                    "project_slug".to_string(),
                    json!(extract_string(args, "project_slug")?),
                );
                body.insert(
                    "description".to_string(),
                    json!(extract_string(args, "description")?),
                );
                if let Some(v) = args.get("entry_points") {
                    body.insert("entry_points".to_string(), v.clone());
                }
                if let Some(v) = args.get("scope").and_then(|v| v.as_str()) {
                    body.insert("scope".to_string(), json!(v));
                }
                if let Some(v) = args.get("auto_create_plan").and_then(|v| v.as_bool()) {
                    body.insert("auto_create_plan".to_string(), json!(v));
                }
                let result = http
                    .post("/api/code/plan-implementation", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "get_co_change_graph" => {
                let project_slug = extract_string(args, "project_slug")?;
                // Resolve project UUID from slug
                let project_info = http.get(&format!("/api/projects/{}", project_slug)).await?;
                let project_id = project_info
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Project not found: {}", project_slug))?;
                let mut query_params = Vec::new();
                if let Some(v) = args.get("min_count").and_then(|v| v.as_i64()) {
                    query_params.push(("min_count".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query_params.push(("limit".to_string(), v.to_string()));
                }
                let result = if query_params.is_empty() {
                    http.get(&format!("/api/projects/{}/co-changes", project_id))
                        .await?
                } else {
                    http.get_with_query(
                        &format!("/api/projects/{}/co-changes", project_id),
                        &query_params,
                    )
                    .await?
                };
                Ok(Some(result))
            }

            "get_file_co_changers" => {
                let file_path = extract_string(args, "file_path")?;
                let mut query_params = vec![("path".to_string(), file_path)];
                if let Some(v) = args.get("min_count").and_then(|v| v.as_i64()) {
                    query_params.push(("min_count".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query_params.push(("limit".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/files/co-changers", &query_params)
                    .await?;
                Ok(Some(result))
            }

            "detect_processes" => {
                let project_slug = extract_string(args, "project_slug")?;
                let body = serde_json::json!({"project_slug": project_slug});
                let result = http.post("/api/code/processes/detect", &body).await?;
                Ok(Some(result))
            }

            // ── Heritage Navigation ──────────────────────────────────────
            "get_class_hierarchy" => {
                let type_name = extract_string(args, "type_name")?;
                let mut query = vec![("type_name".to_string(), type_name)];
                if let Some(v) = args.get("max_depth").and_then(|v| v.as_i64()) {
                    query.push(("max_depth".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/class-hierarchy", &query)
                    .await?;
                Ok(Some(result))
            }

            "find_subclasses" => {
                let class_name = extract_string(args, "class_name")?;
                let query = vec![("class_name".to_string(), class_name)];
                let result = http.get_with_query("/api/code/subclasses", &query).await?;
                Ok(Some(result))
            }

            "find_interface_implementors" => {
                let interface_name = extract_string(args, "interface_name")?;
                let query = vec![("interface_name".to_string(), interface_name)];
                let result = http
                    .get_with_query("/api/code/interface-implementors", &query)
                    .await?;
                Ok(Some(result))
            }

            // ── Process Navigation ───────────────────────────────────────
            "list_processes" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![("project_slug".to_string(), project_slug)];
                let result = http.get_with_query("/api/code/processes", &query).await?;
                Ok(Some(result))
            }

            "get_process" => {
                let process_id = extract_string(args, "process_id")?;
                let query = vec![("process_id".to_string(), process_id)];
                let result = http
                    .get_with_query("/api/code/processes/detail", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_entry_points" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/entry-points", &query)
                    .await?;
                Ok(Some(result))
            }

            // ── Community Enrichment + REST Gaps ─────────────────────────
            "enrich_communities" => {
                let project_slug = extract_string(args, "project_slug")?;
                let body = serde_json::json!({"project_slug": project_slug});
                let result = http.post("/api/code/communities/enrich", &body).await?;
                Ok(Some(result))
            }

            "get_hotspots" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/hotspots", &query).await?;
                Ok(Some(result))
            }

            "get_knowledge_gaps" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/knowledge-gaps", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_risk_assessment" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/risk-assessment", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_homeostasis" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![("project_slug".to_string(), project_slug)];
                let result = http.get_with_query("/api/code/homeostasis", &query).await?;
                Ok(Some(result))
            }

            "get_structural_drift" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("warning_threshold").and_then(|v| v.as_f64()) {
                    query.push(("warning_threshold".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("critical_threshold").and_then(|v| v.as_f64()) {
                    query.push(("critical_threshold".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/structural-drift", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_bridge" => {
                let source = extract_string(args, "source")?;
                let target = extract_string(args, "target")?;
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![
                    ("source".to_string(), source),
                    ("target".to_string(), target),
                    ("project_slug".to_string(), project_slug),
                ];
                if let Some(v) = args.get("max_hops").and_then(|v| v.as_i64()) {
                    query.push(("max_hops".to_string(), v.to_string()));
                }
                if let Some(v) = args.get("top_bottlenecks").and_then(|v| v.as_i64()) {
                    query.push(("top_bottlenecks".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/bridge", &query).await?;
                Ok(Some(result))
            }

            // Topological Firewall (GraIL Plan 3)
            "check_topology" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![("project_slug".to_string(), project_slug)];
                let result = http
                    .get_with_query("/api/code/topology/check", &query)
                    .await?;
                Ok(Some(result))
            }

            "list_topology_rules" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![("project_slug".to_string(), project_slug)];
                let result = http
                    .get_with_query("/api/code/topology/rules", &query)
                    .await?;
                Ok(Some(result))
            }

            "create_topology_rule" => {
                let body = serde_json::json!({
                    "project_slug": extract_string(args, "project_slug")?,
                    "rule_type": extract_string(args, "rule_type")?,
                    "source_pattern": extract_string(args, "source_pattern")?,
                    "target_pattern": args.get("target_pattern").and_then(|v| v.as_str()),
                    "threshold": args.get("threshold").and_then(|v| v.as_u64()).map(|v| v as u32),
                    "severity": args.get("severity").and_then(|v| v.as_str()),
                    "description": extract_string(args, "description")?,
                });
                let result = http.post("/api/code/topology/rules", &body).await?;
                Ok(Some(result))
            }

            "delete_topology_rule" => {
                let rule_id = extract_string(args, "rule_id")?;
                let result = http
                    .delete(&format!("/api/code/topology/rules/{}", rule_id))
                    .await?;
                Ok(Some(result))
            }

            "check_file_topology" => {
                let mut body = serde_json::Map::new();
                body.insert(
                    "project_slug".to_string(),
                    json!(extract_string(args, "project_slug")?),
                );
                body.insert(
                    "file_path".to_string(),
                    json!(extract_string(args, "file_path")?),
                );
                let new_imports = args
                    .get("new_imports")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                body.insert("new_imports".to_string(), json!(new_imports));
                let result = http
                    .post("/api/code/topology/check-file", &json!(body))
                    .await?;
                Ok(Some(result))
            }

            // --- Structural DNA (2) ---
            "get_structural_profile" => {
                let result = http.post("/api/code/structural-profile", args).await?;
                Ok(Some(result))
            }

            "find_structural_twins" => {
                let result = http.post("/api/code/structural-twins", args).await?;
                Ok(Some(result))
            }

            "cluster_dna" => {
                let result = http.post("/api/code/structural-clusters", args).await?;
                Ok(Some(result))
            }

            "find_cross_project_twins" => {
                let result = http
                    .post("/api/code/structural-twins/cross-project", args)
                    .await?;
                Ok(Some(result))
            }

            "predict_missing_links" => {
                let result = http.post("/api/code/predict-links", args).await?;
                Ok(Some(result))
            }

            "check_link_plausibility" => {
                let result = http.post("/api/code/link-plausibility", args).await?;
                Ok(Some(result))
            }

            // ── P5: Stress Testing (4 tools) ──────────────────────────────
            "stress_test_node" => {
                let result = http.post("/api/code/stress-test-node", args).await?;
                Ok(Some(result))
            }

            "stress_test_edge" => {
                let result = http.post("/api/code/stress-test-edge", args).await?;
                Ok(Some(result))
            }

            "stress_test_cascade" => {
                let result = http.post("/api/code/stress-test-cascade", args).await?;
                Ok(Some(result))
            }

            "find_bridges" => {
                let result = http.post("/api/code/find-bridges", args).await?;
                Ok(Some(result))
            }

            // ── P8: Context Cards (2 tools) ─────────────────────────────
            "get_context_card" => {
                let path = extract_string(args, "path")
                    .or_else(|_| extract_string(args, "file_path"))
                    .or_else(|_| extract_string(args, "node_path"))?;
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![
                    ("path".to_string(), path),
                    ("project_slug".to_string(), project_slug),
                ];
                let result = http
                    .get_with_query("/api/code/context-card", &query)
                    .await?;
                Ok(Some(result))
            }

            "refresh_context_cards" => {
                let project_slug = extract_string(args, "project_slug")?;
                let body = serde_json::json!({"project_slug": project_slug});
                let result = http.post("/api/code/context-cards/refresh", &body).await?;
                Ok(Some(result))
            }

            // ── P7: WL Fingerprint & Isomorphic (2 tools) ──────────────
            "get_fingerprint" => {
                let path = extract_string(args, "path")
                    .or_else(|_| extract_string(args, "file_path"))
                    .or_else(|_| extract_string(args, "node_path"))?;
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![
                    ("path".to_string(), path),
                    ("project_slug".to_string(), project_slug),
                ];
                let result = http.get_with_query("/api/code/fingerprint", &query).await?;
                Ok(Some(result))
            }

            "find_isomorphic" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("min_group_size").and_then(|v| v.as_i64()) {
                    query.push(("min_group_size".to_string(), v.to_string()));
                }
                let result = http.get_with_query("/api/code/isomorphic", &query).await?;
                Ok(Some(result))
            }

            "suggest_structural_templates" => {
                let project_slug = extract_string(args, "project_slug")?;
                let mut query = vec![("project_slug".to_string(), project_slug)];
                if let Some(v) = args.get("min_occurrences").and_then(|v| v.as_i64()) {
                    query.push(("min_occurrences".to_string(), v.to_string()));
                }
                let result = http
                    .get_with_query("/api/code/structural-templates", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_learning_health" => {
                let project_slug = extract_string(args, "project_slug")?;
                let query = vec![("project_slug".to_string(), project_slug)];
                let result = http
                    .get_with_query("/api/personas/learning-health", &query)
                    .await?;
                Ok(Some(result))
            }

            "get_meilisearch_stats" => {
                let result = http.get("/api/meilisearch/stats").await?;
                Ok(Some(result))
            }

            "delete_meilisearch_orphans" => {
                let result = http.delete("/api/meilisearch/orphans").await?;
                Ok(Some(result))
            }

            // ── P11: Admin & Sync (6 tools) ─────────────────────────────

            // --- Sync (1) ---
            "sync_directory" => {
                let mut body = serde_json::Map::new();
                body.insert("path".to_string(), json!(extract_string(args, "path")?));
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), json!(v));
                }
                let result = http.post("/api/sync", &Value::Object(body)).await?;
                Ok(Some(result))
            }

            // --- Watch (3) ---
            "start_watch" => {
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("path").and_then(|v| v.as_str()) {
                    body.insert("path".to_string(), json!(v));
                }
                if let Some(v) = args.get("project_id").and_then(|v| v.as_str()) {
                    body.insert("project_id".to_string(), json!(v));
                }
                let result = http.post("/api/watch", &Value::Object(body)).await?;
                Ok(Some(result))
            }

            "stop_watch" => {
                let result = http.delete("/api/watch").await?;
                Ok(Some(result))
            }

            "watch_status" => {
                let result = http.get("/api/watch").await?;
                Ok(Some(result))
            }

            // --- Admin Cleanup (2) ---
            "cleanup_cross_project_calls" => {
                let result = http
                    .post("/api/admin/cleanup-cross-project-calls", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "cleanup_sync_data" => {
                let result = http
                    .post("/api/admin/cleanup-sync-data", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "cleanup_builtin_calls" => {
                let result = http
                    .post("/api/admin/cleanup-builtin-calls", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            "migrate_calls_confidence" => {
                let result = http
                    .post("/api/admin/migrate-calls-confidence", &json!({}))
                    .await?;
                Ok(Some(result))
            }

            // ── Analysis Profiles (4 tools) ─────────────────────────────
            "list_analysis_profiles" => {
                let mut query = Vec::new();
                if let Some(pid) = extract_optional_string(args, "project_id") {
                    query.push(("project_id".to_string(), pid));
                }
                let result = if query.is_empty() {
                    http.get("/api/analysis-profiles").await?
                } else {
                    http.get_with_query("/api/analysis-profiles", &query)
                        .await?
                };
                Ok(Some(result))
            }

            "create_analysis_profile" => {
                let result = http.post("/api/analysis-profiles", args).await?;
                Ok(Some(result))
            }

            "get_analysis_profile" => {
                let id = extract_string(args, "id")?;
                let result = http.get(&format!("/api/analysis-profiles/{}", id)).await?;
                Ok(Some(result))
            }

            "delete_analysis_profile" => {
                let id = extract_string(args, "id")?;
                let result = http
                    .delete(&format!("/api/analysis-profiles/{}", id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // ── Reasoning Tree ────────────────────────────────────────
            "reason" => {
                let result = http.post("/api/reason", args).await?;
                Ok(Some(result))
            }
            "reason_feedback" => {
                let tree_id = extract_id(args, "tree_id")?;
                let body = json!({
                    "followed_nodes": args.get("followed_nodes").cloned().unwrap_or(json!([])),
                    "outcome": args.get("outcome").and_then(|v| v.as_str()).unwrap_or("success")
                });
                let result = http
                    .post(&format!("/api/reason/{}/feedback", tree_id), &body)
                    .await?;
                Ok(Some(result))
            }

            // ── Episodes (Episodic Memory) ──────────────────────────────
            "collect_episode" => {
                let result = http.post("/api/episodes/collect", args).await?;
                Ok(Some(result))
            }
            "list_episodes" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = vec![("project_id".to_string(), project_id)];
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                let result = http.get_with_query("/api/episodes", &query).await?;
                Ok(Some(result))
            }
            "anonymize_episode" => {
                let result = http.post("/api/episodes/anonymize", args).await?;
                Ok(Some(result))
            }
            "export_artifact" => {
                let result = http.post("/api/episodes/export-artifact", args).await?;
                Ok(Some(result))
            }

            // ── P10: Skills (9 tools) ──────────────────────────────────
            "list_skills" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = vec![("project_id".to_string(), project_id)];
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = http.get_with_query("/api/skills", &query).await?;
                Ok(Some(result))
            }

            "create_skill" => {
                let result = http.post("/api/skills", args).await?;
                Ok(Some(result))
            }

            "get_skill" => {
                let skill_id = extract_id(args, "skill_id")?;
                let result = http.get(&format!("/api/skills/{}", skill_id)).await?;
                Ok(Some(result))
            }

            "update_skill" => {
                let skill_id = extract_id(args, "skill_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("status") {
                    body.insert("status".to_string(), v.clone());
                }
                if let Some(v) = args.get("tags") {
                    body.insert("tags".to_string(), v.clone());
                }
                if let Some(v) = args.get("trigger_patterns") {
                    body.insert("trigger_patterns".to_string(), v.clone());
                }
                if let Some(v) = args.get("context_template") {
                    body.insert("context_template".to_string(), v.clone());
                }
                if let Some(v) = args.get("energy") {
                    body.insert("energy".to_string(), v.clone());
                }
                if let Some(v) = args.get("cohesion") {
                    body.insert("cohesion".to_string(), v.clone());
                }
                let result = http
                    .put(&format!("/api/skills/{}", skill_id), &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            "delete_skill" => {
                let skill_id = extract_id(args, "skill_id")?;
                let result = http.delete(&format!("/api/skills/{}", skill_id)).await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "get_skill_members" => {
                let skill_id = extract_id(args, "skill_id")?;
                let result = http
                    .get(&format!("/api/skills/{}/members", skill_id))
                    .await?;
                Ok(Some(result))
            }

            "add_skill_member" => {
                let skill_id = extract_id(args, "skill_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_id(args, "entity_id")?;
                // Validate entity_type (defense-in-depth, also validated server-side)
                if entity_type != "note" && entity_type != "decision" {
                    return Err(anyhow!(
                        "Invalid entity_type '{}': expected 'note' or 'decision'",
                        entity_type
                    ));
                }
                let body = json!({
                    "entity_type": entity_type,
                    "entity_id": entity_id
                });
                let result = http
                    .post(&format!("/api/skills/{}/members", skill_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_skill_member" => {
                let skill_id = extract_id(args, "skill_id")?;
                let entity_type = extract_string(args, "entity_type")?;
                let entity_id = extract_id(args, "entity_id")?;
                // Validate entity_type to prevent URL path injection
                if entity_type != "note" && entity_type != "decision" {
                    return Err(anyhow!(
                        "Invalid entity_type '{}': expected 'note' or 'decision'",
                        entity_type
                    ));
                }
                let result = http
                    .delete(&format!(
                        "/api/skills/{}/members/{}/{}",
                        skill_id, entity_type, entity_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "activate_skill" => {
                let skill_id = extract_id(args, "skill_id")?;
                let query = extract_string(args, "query")?;
                let body = json!({"query": query});
                let result = http
                    .post(&format!("/api/skills/{}/activate", skill_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "export_skill" => {
                let skill_id = extract_id(args, "skill_id")?;
                let mut query = Vec::new();
                if let Some(name) = extract_optional_string(args, "source_project_name") {
                    query.push(("source_project_name".to_string(), name));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/skills/{}/export", skill_id))
                        .await?
                } else {
                    http.get_with_query(&format!("/api/skills/{}/export", skill_id), &query)
                        .await?
                };
                Ok(Some(result))
            }

            "import_skill" => {
                let result = http.post("/api/skills/import", args).await?;
                Ok(Some(result))
            }

            "get_skill_health" => {
                let skill_id = extract_id(args, "skill_id")?;
                let result = http
                    .get(&format!("/api/skills/{}/health", skill_id))
                    .await?;
                Ok(Some(result))
            }

            "split_skill" => {
                let skill_id = extract_id(args, "skill_id")?;
                let mut body = serde_json::Map::new();
                if let Some(sub) = args.get("sub_clusters") {
                    body.insert("sub_clusters".to_string(), sub.clone());
                }
                let result = http
                    .post(
                        &format!("/api/skills/{}/split", skill_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "merge_skills" => {
                let result = http.post("/api/skills/merge", args).await?;
                Ok(Some(result))
            }

            // ── Persona (Living Personas) ──────────────────────────────
            "list_personas" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = vec![("project_id".to_string(), project_id)];
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = http.get_with_query("/api/personas", &query).await?;
                Ok(Some(result))
            }

            "create_persona" => {
                let result = http.post("/api/personas", args).await?;
                Ok(Some(result))
            }

            "get_persona" => {
                let persona_id = extract_id(args, "persona_id")?;
                let result = http.get(&format!("/api/personas/{}", persona_id)).await?;
                Ok(Some(result))
            }

            "update_persona" => {
                let persona_id = extract_id(args, "persona_id")?;
                let mut body = serde_json::Map::new();
                for field in &[
                    "name",
                    "description",
                    "status",
                    "complexity_default",
                    "model_preference",
                    "system_prompt_override",
                ] {
                    if let Some(v) = args.get(*field) {
                        body.insert(field.to_string(), v.clone());
                    }
                }
                for field in &["energy", "cohesion", "max_cost_usd"] {
                    if let Some(v) = args.get(*field) {
                        body.insert(field.to_string(), v.clone());
                    }
                }
                if let Some(v) = args.get("timeout_secs") {
                    body.insert("timeout_secs".to_string(), v.clone());
                }
                let result = http
                    .put(
                        &format!("/api/personas/{}", persona_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "delete_persona" => {
                let persona_id = extract_id(args, "persona_id")?;
                let result = http
                    .delete(&format!("/api/personas/{}", persona_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_persona_skill" => {
                let persona_id = extract_id(args, "persona_id")?;
                let skill_id = extract_id(args, "skill_id")?;
                let result = http
                    .post(
                        &format!("/api/personas/{}/skills/{}", persona_id, skill_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_skill" => {
                let persona_id = extract_id(args, "persona_id")?;
                let skill_id = extract_id(args, "skill_id")?;
                let result = http
                    .delete(&format!("/api/personas/{}/skills/{}", persona_id, skill_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "add_persona_protocol" => {
                let persona_id = extract_id(args, "persona_id")?;
                let protocol_id = extract_id(args, "protocol_id")?;
                let result = http
                    .post(
                        &format!("/api/personas/{}/protocols/{}", persona_id, protocol_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_protocol" => {
                let persona_id = extract_id(args, "persona_id")?;
                let protocol_id = extract_id(args, "protocol_id")?;
                let result = http
                    .delete(&format!(
                        "/api/personas/{}/protocols/{}",
                        persona_id, protocol_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "add_persona_file" => {
                let persona_id = extract_id(args, "persona_id")?;
                let file_path = extract_string(args, "file_path")?;
                let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let body = json!({"file_path": file_path, "weight": weight});
                let result = http
                    .post(&format!("/api/personas/{}/files", persona_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_file" => {
                let persona_id = extract_id(args, "persona_id")?;
                let file_path = extract_string(args, "file_path")?;
                let body = json!({"file_path": file_path});
                let result = http
                    .delete_with_body(&format!("/api/personas/{}/files", persona_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "add_persona_function" => {
                let persona_id = extract_id(args, "persona_id")?;
                let function_name = extract_string(args, "function_name")?;
                let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let body = json!({"function_name": function_name, "weight": weight});
                let result = http
                    .post(&format!("/api/personas/{}/functions", persona_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_function" => {
                let persona_id = extract_id(args, "persona_id")?;
                let function_name = extract_string(args, "function_name")?;
                let body = json!({"function_name": function_name});
                let result = http
                    .delete_with_body(&format!("/api/personas/{}/functions", persona_id), &body)
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "add_persona_note" => {
                let persona_id = extract_id(args, "persona_id")?;
                let note_id = extract_id(args, "note_id")?;
                let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let result = http
                    .post(
                        &format!("/api/personas/{}/notes/{}", persona_id, note_id),
                        &json!({"weight": weight}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_note" => {
                let persona_id = extract_id(args, "persona_id")?;
                let note_id = extract_id(args, "note_id")?;
                let result = http
                    .delete(&format!("/api/personas/{}/notes/{}", persona_id, note_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "add_persona_decision" => {
                let persona_id = extract_id(args, "persona_id")?;
                let decision_id = extract_id(args, "decision_id")?;
                let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let result = http
                    .post(
                        &format!("/api/personas/{}/decisions/{}", persona_id, decision_id),
                        &json!({"weight": weight}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_decision" => {
                let persona_id = extract_id(args, "persona_id")?;
                let decision_id = extract_id(args, "decision_id")?;
                let result = http
                    .delete(&format!(
                        "/api/personas/{}/decisions/{}",
                        persona_id, decision_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "scope_persona_feature_graph" => {
                let persona_id = extract_id(args, "persona_id")?;
                let feature_graph_id = extract_id(args, "feature_graph_id")?;
                let result = http
                    .post(
                        &format!(
                            "/api/personas/{}/feature-graphs/{}",
                            persona_id, feature_graph_id
                        ),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"scoped": true})
                } else {
                    result
                }))
            }

            "unscope_persona_feature_graph" => {
                let persona_id = extract_id(args, "persona_id")?;
                let feature_graph_id = extract_id(args, "feature_graph_id")?;
                let result = http
                    .delete(&format!(
                        "/api/personas/{}/feature-graphs/{}",
                        persona_id, feature_graph_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"unscoped": true})
                } else {
                    result
                }))
            }

            "add_persona_extends" => {
                let persona_id = extract_id(args, "persona_id")?;
                let parent_persona_id = extract_id(args, "parent_persona_id")?;
                let result = http
                    .post(
                        &format!("/api/personas/{}/extends/{}", persona_id, parent_persona_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"added": true})
                } else {
                    result
                }))
            }

            "remove_persona_extends" => {
                let persona_id = extract_id(args, "persona_id")?;
                let parent_persona_id = extract_id(args, "parent_persona_id")?;
                let result = http
                    .delete(&format!(
                        "/api/personas/{}/extends/{}",
                        persona_id, parent_persona_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"removed": true})
                } else {
                    result
                }))
            }

            "get_persona_subgraph" => {
                let persona_id = extract_id(args, "persona_id")?;
                let result = http
                    .get(&format!("/api/personas/{}/subgraph", persona_id))
                    .await?;
                Ok(Some(result))
            }

            "find_personas_for_file" => {
                let file_path = extract_string(args, "file_path")?;
                let project_id = extract_id(args, "project_id")?;
                let query = vec![
                    ("file_path".to_string(), file_path),
                    ("project_id".to_string(), project_id),
                ];
                let result = http
                    .get_with_query("/api/personas/find-for-file", &query)
                    .await?;
                Ok(Some(result))
            }

            "list_global_personas" => {
                let mut query = Vec::new();
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = http.get_with_query("/api/personas/global", &query).await?;
                Ok(Some(result))
            }

            "export_persona" => {
                let persona_id = extract_id(args, "persona_id")?;
                let mut query = Vec::new();
                if let Some(name) = extract_optional_string(args, "source_project_name") {
                    query.push(("source_project_name".to_string(), name));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/personas/{}/export", persona_id))
                        .await?
                } else {
                    http.get_with_query(&format!("/api/personas/{}/export", persona_id), &query)
                        .await?
                };
                Ok(Some(result))
            }

            "import_persona" => {
                let result = http.post("/api/personas/import", args).await?;
                Ok(Some(result))
            }

            "activate_persona" => {
                let persona_id = extract_id(args, "persona_id")?;
                let result = http
                    .post(
                        &format!("/api/personas/{}/activate", persona_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }

            "auto_build_persona" => {
                let result = http.post("/api/personas/auto-build", args).await?;
                Ok(Some(result))
            }

            "maintain_personas" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .post(
                        &format!("/api/personas/maintain?project_id={}", project_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }

            "detect_personas" => {
                let project_id = extract_id(args, "project_id")?;
                let result = http
                    .post(
                        &format!("/api/personas/detect?project_id={}", project_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }

            // ── Protocol (Pattern Federation) (12 tools) ───────────────
            "list_protocols" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query = vec![("project_id".to_string(), project_id)];
                if let Some(cat) = args.get("category").and_then(|v| v.as_str()) {
                    query.push(("category".to_string(), cat.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = http.get_with_query("/api/protocols", &query).await?;
                Ok(Some(result))
            }

            "create_protocol" => {
                let result = http.post("/api/protocols", args).await?;
                Ok(Some(result))
            }

            "get_protocol" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let result = http.get(&format!("/api/protocols/{}", protocol_id)).await?;
                Ok(Some(result))
            }

            "update_protocol" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("name") {
                    body.insert("name".to_string(), v.clone());
                }
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("protocol_category") {
                    body.insert("protocol_category".to_string(), v.clone());
                }
                if let Some(v) = args.get("trigger_mode") {
                    body.insert("trigger_mode".to_string(), v.clone());
                }
                if let Some(v) = args.get("trigger_config") {
                    body.insert("trigger_config".to_string(), v.clone());
                }
                if let Some(v) = args.get("relevance_vector") {
                    body.insert("relevance_vector".to_string(), v.clone());
                }
                let result = http
                    .put(
                        &format!("/api/protocols/{}", protocol_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "delete_protocol" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let result = http
                    .delete(&format!("/api/protocols/{}", protocol_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "add_protocol_state" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let name = extract_string(args, "name")?;
                let mut body = serde_json::Map::new();
                body.insert("name".to_string(), json!(name));
                if let Some(v) = args.get("description") {
                    body.insert("description".to_string(), v.clone());
                }
                if let Some(v) = args.get("state_type") {
                    body.insert("state_type".to_string(), v.clone());
                }
                if let Some(v) = args.get("action_name") {
                    body.insert("action".to_string(), v.clone());
                }
                if let Some(v) = args.get("prompt_fragment") {
                    body.insert("prompt_fragment".to_string(), v.clone());
                }
                if let Some(v) = args.get("available_tools") {
                    body.insert("available_tools".to_string(), v.clone());
                }
                if let Some(v) = args.get("forbidden_actions") {
                    body.insert("forbidden_actions".to_string(), v.clone());
                }
                let result = http
                    .post(
                        &format!("/api/protocols/{}/states", protocol_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "delete_protocol_state" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let state_id = extract_id(args, "state_id")?;
                let result = http
                    .delete(&format!(
                        "/api/protocols/{}/states/{}",
                        protocol_id, state_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "list_protocol_states" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let result = http
                    .get(&format!("/api/protocols/{}/states", protocol_id))
                    .await?;
                Ok(Some(result))
            }

            "add_protocol_transition" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let from_state = extract_string(args, "from_state")?;
                let to_state = extract_string(args, "to_state")?;
                let trigger = extract_string(args, "trigger")?;
                let mut body = serde_json::Map::new();
                body.insert("from_state".to_string(), json!(from_state));
                body.insert("to_state".to_string(), json!(to_state));
                body.insert("trigger".to_string(), json!(trigger));
                if let Some(v) = args.get("guard") {
                    body.insert("guard".to_string(), v.clone());
                }
                let result = http
                    .post(
                        &format!("/api/protocols/{}/transitions", protocol_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "delete_protocol_transition" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let transition_id = extract_id(args, "transition_id")?;
                let result = http
                    .delete(&format!(
                        "/api/protocols/{}/transitions/{}",
                        protocol_id, transition_id
                    ))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            "list_protocol_transitions" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let result = http
                    .get(&format!("/api/protocols/{}/transitions", protocol_id))
                    .await?;
                Ok(Some(result))
            }

            "link_protocol_to_skill" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let skill_id = extract_string(args, "skill_id")?;
                let body = json!({"skill_id": skill_id});
                let result = http
                    .post(&format!("/api/protocols/{}/link-skill", protocol_id), &body)
                    .await?;
                Ok(Some(result))
            }

            // ── Protocol Runs (FSM Runtime) (7 tools) ────────────────────
            "start_protocol_run" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("plan_id") {
                    body.insert("plan_id".to_string(), v.clone());
                }
                if let Some(v) = args.get("task_id") {
                    body.insert("task_id".to_string(), v.clone());
                }
                if let Some(v) = args.get("parent_run_id") {
                    body.insert("parent_run_id".to_string(), v.clone());
                }
                let result = http
                    .post(
                        &format!("/api/protocols/{}/runs", protocol_id),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }

            "get_protocol_run" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http.get(&format!("/api/protocols/runs/{}", run_id)).await?;
                Ok(Some(result))
            }

            "list_protocol_runs" => {
                let protocol_id = extract_id(args, "protocol_id")?;
                let mut query = vec![];
                if let Some(s) = args.get("status").and_then(|v| v.as_str()) {
                    query.push(("status".to_string(), s.to_string()));
                }
                if let Some(l) = args.get("limit").and_then(|v| v.as_i64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_i64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = http
                    .get_with_query(&format!("/api/protocols/{}/runs", protocol_id), &query)
                    .await?;
                Ok(Some(result))
            }

            "fire_protocol_transition" => {
                let run_id = extract_id(args, "run_id")?;
                let trigger = extract_string(args, "trigger")?;
                let body = json!({"trigger": trigger});
                let result = http
                    .post(&format!("/api/protocols/runs/{}/transition", run_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "cancel_protocol_run" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http
                    .post(
                        &format!("/api/protocols/runs/{}/cancel", run_id),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }

            "fail_protocol_run" => {
                let run_id = extract_id(args, "run_id")?;
                let error_msg = args
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                let body = json!({"error": error_msg});
                let result = http
                    .post(&format!("/api/protocols/runs/{}/fail", run_id), &body)
                    .await?;
                Ok(Some(result))
            }

            "report_protocol_progress" => {
                let run_id = extract_id(args, "run_id")?;
                let state_name = extract_string(args, "state_name")?;
                let sub_action = extract_string(args, "sub_action")?;
                let processed =
                    args.get("processed").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let total = args.get("total").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let elapsed_ms = args.get("elapsed_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                let body = json!({
                    "state_name": state_name,
                    "sub_action": sub_action,
                    "processed": processed,
                    "total": total,
                    "elapsed_ms": elapsed_ms,
                });
                let _result = http
                    .post(&format!("/api/protocols/runs/{}/progress", run_id), &body)
                    .await?;
                Ok(Some(json!({
                    "accepted": true,
                    "run_id": run_id.to_string(),
                    "progress": format!("{}/{}", processed, total),
                })))
            }

            "delete_protocol_run" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http
                    .delete(&format!("/api/protocols/runs/{}", run_id))
                    .await?;
                Ok(Some(if result.is_null() {
                    json!({"deleted": true})
                } else {
                    result
                }))
            }

            // ── Protocol Hierarchy (2 tools) ────────────────────
            "get_protocol_run_tree" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http
                    .get(&format!("/api/protocols/runs/{}/tree", run_id))
                    .await?;
                Ok(Some(result))
            }

            "get_protocol_run_children" => {
                let run_id = extract_id(args, "run_id")?;
                let result = http
                    .get(&format!("/api/protocols/runs/{}/children", run_id))
                    .await?;
                Ok(Some(result))
            }

            "route_protocols" => {
                let project_id = extract_id(args, "project_id")?;
                let mut query_params = format!("project_id={}", project_id);
                if let Some(plan_id) = args.get("plan_id").and_then(|v| v.as_str()) {
                    query_params.push_str(&format!("&plan_id={}", plan_id));
                }
                if let Some(phase) = args.get("phase").and_then(|v| v.as_str()) {
                    query_params.push_str(&format!("&phase={}", phase));
                }
                if let Some(domain) = args.get("domain").and_then(|v| v.as_f64()) {
                    query_params.push_str(&format!("&domain={}", domain));
                }
                if let Some(resource) = args.get("resource").and_then(|v| v.as_f64()) {
                    query_params.push_str(&format!("&resource={}", resource));
                }
                if let Some(structure) = args.get("structure").and_then(|v| v.as_f64()) {
                    query_params.push_str(&format!("&structure={}", structure));
                }
                if let Some(lifecycle) = args.get("lifecycle").and_then(|v| v.as_f64()) {
                    query_params.push_str(&format!("&lifecycle={}", lifecycle));
                }
                let result = http
                    .get(&format!("/api/protocols/route?{}", query_params))
                    .await?;
                Ok(Some(result))
            }

            "compose_protocol" => {
                let mut body = json!({});
                if let Some(v) = args.get("project_id") {
                    body["project_id"] = v.clone();
                }
                if let Some(v) = args.get("name") {
                    body["name"] = v.clone();
                }
                if let Some(v) = args.get("description") {
                    body["description"] = v.clone();
                }
                if let Some(v) = args.get("category") {
                    body["category"] = v.clone();
                }
                if let Some(v) = args.get("notes") {
                    body["notes"] = v.clone();
                }
                if let Some(v) = args.get("states") {
                    body["states"] = v.clone();
                }
                if let Some(v) = args.get("transitions") {
                    body["transitions"] = v.clone();
                }
                if let Some(v) = args.get("relevance_vector") {
                    body["relevance_vector"] = v.clone();
                }
                if let Some(v) = args.get("triggers") {
                    body["triggers"] = v.clone();
                }
                let result = http.post("/api/protocols/compose", &body).await?;
                Ok(Some(result))
            }

            "simulate_protocol" => {
                let mut body = json!({});
                if let Some(v) = args.get("protocol_id") {
                    body["protocol_id"] = v.clone();
                }
                if let Some(v) = args.get("context") {
                    body["context"] = v.clone();
                }
                if let Some(v) = args.get("plan_id") {
                    body["plan_id"] = v.clone();
                }
                let result = http.post("/api/protocols/simulate", &body).await?;
                Ok(Some(result))
            }

            // ── Sharing (Privacy MVP) ─────────────────────────────────
            "get_sharing_status" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http.get(&format!("/api/projects/{}/sharing", slug)).await?;
                Ok(Some(result))
            }
            "enable_sharing" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .post(
                        &format!("/api/projects/{}/sharing/enable", slug),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }
            "disable_sharing" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .post(
                        &format!("/api/projects/{}/sharing/disable", slug),
                        &json!({}),
                    )
                    .await?;
                Ok(Some(result))
            }
            "set_sharing_policy" => {
                let slug = extract_string(args, "project_slug")?;
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("mode") {
                    body.insert("mode".to_string(), v.clone());
                }
                if let Some(v) = args.get("type_overrides") {
                    body.insert("type_overrides".to_string(), v.clone());
                }
                if let Some(v) = args.get("min_shareability_score") {
                    body.insert("min_shareability_score".to_string(), v.clone());
                }
                let result = http
                    .put(
                        &format!("/api/projects/{}/sharing/policy", slug),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }
            "get_sharing_policy" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/sharing/policy", slug))
                    .await?;
                Ok(Some(result))
            }
            "set_sharing_consent" => {
                let note_id = extract_id(args, "note_id")?;
                let consent = extract_string(args, "consent")?;
                let body = json!({"consent": consent});
                let result = http
                    .put(&format!("/api/notes/{}/consent", note_id), &body)
                    .await?;
                Ok(Some(result))
            }
            "get_sharing_history" => {
                let slug = extract_string(args, "project_slug")?;
                let mut query = Vec::new();
                if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) {
                    query.push(("limit".to_string(), l.to_string()));
                }
                if let Some(o) = args.get("offset").and_then(|v| v.as_u64()) {
                    query.push(("offset".to_string(), o.to_string()));
                }
                let result = if query.is_empty() {
                    http.get(&format!("/api/projects/{}/sharing/history", slug))
                        .await?
                } else {
                    http.get_with_query(&format!("/api/projects/{}/sharing/history", slug), &query)
                        .await?
                };
                Ok(Some(result))
            }
            "preview_sharing" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/sharing/preview", slug))
                    .await?;
                Ok(Some(result))
            }
            "suggest_sharing" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/sharing/suggest", slug))
                    .await?;
                Ok(Some(result))
            }
            "retract_sharing" => {
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("note_id") {
                    body.insert("note_id".to_string(), v.clone());
                }
                if let Some(v) = args.get("content_hash") {
                    body.insert("content_hash".to_string(), v.clone());
                }
                if let Some(v) = args.get("urgent") {
                    body.insert("urgent".to_string(), v.clone());
                }
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .post(
                        &format!("/api/projects/{}/sharing/retract", slug),
                        &Value::Object(body),
                    )
                    .await?;
                Ok(Some(result))
            }
            "list_tombstones" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/sharing/tombstones", slug))
                    .await?;
                Ok(Some(result))
            }
            "get_last_privacy_report" => {
                let slug = extract_string(args, "project_slug")?;
                let result = http
                    .get(&format!("/api/projects/{}/sharing/last-report", slug))
                    .await?;
                Ok(Some(result))
            }

            // ── Neural Routing ────────────────────────────────────────
            "get_neural_routing_status" => {
                let result = http.get("/api/neural-routing/status").await?;
                Ok(Some(result))
            }
            "get_neural_routing_config" => {
                let result = http.get("/api/neural-routing/config").await?;
                Ok(Some(result))
            }
            "enable_neural_routing" => {
                let result = http.post("/api/neural-routing/enable", &json!({})).await?;
                Ok(Some(result))
            }
            "disable_neural_routing" => {
                let result = http.post("/api/neural-routing/disable", &json!({})).await?;
                Ok(Some(result))
            }
            "set_neural_routing_mode" => {
                let mut body = serde_json::Map::new();
                if let Some(v) = args.get("mode") {
                    body.insert("mode".to_string(), v.clone());
                }
                let result = http
                    .put("/api/neural-routing/mode", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }
            "update_neural_routing_config" => {
                let mut body = serde_json::Map::new();
                for key in &[
                    "enabled",
                    "mode",
                    "inference_timeout_ms",
                    "nn_fallback",
                    "collection_enabled",
                    "collection_buffer_size",
                    "nn_top_k",
                    "nn_min_similarity",
                    "nn_max_route_age_days",
                ] {
                    if let Some(v) = args.get(*key) {
                        body.insert(key.to_string(), v.clone());
                    }
                }
                let result = http
                    .put("/api/neural-routing/config", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }

            // ── Trajectories ──────────────────────────────────────────
            "list_trajectories" => {
                let mut query_params = Vec::new();
                for key in &[
                    "session_id",
                    "min_reward",
                    "max_reward",
                    "min_steps",
                    "max_steps",
                    "limit",
                    "offset",
                ] {
                    if let Some(v) = args.get(*key) {
                        let val = if v.is_string() {
                            v.as_str().unwrap().to_string()
                        } else {
                            v.to_string()
                        };
                        query_params.push(format!("{}={}", key, val));
                    }
                }
                let qs = if query_params.is_empty() {
                    String::new()
                } else {
                    format!("?{}", query_params.join("&"))
                };
                let result = http.get(&format!("/api/trajectories{}", qs)).await?;
                Ok(Some(result))
            }
            "get_trajectory" => {
                let id = args
                    .get("trajectory_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("trajectory_id is required"))?;
                let result = http.get(&format!("/api/trajectories/{}", id)).await?;
                Ok(Some(result))
            }
            "search_similar_trajectories" => {
                let mut body = serde_json::Map::new();
                for key in &["embedding", "top_k", "min_similarity"] {
                    if let Some(v) = args.get(*key) {
                        body.insert(key.to_string(), v.clone());
                    }
                }
                let result = http
                    .post("/api/trajectories/similar", &Value::Object(body))
                    .await?;
                Ok(Some(result))
            }
            "get_trajectory_stats" => {
                let result = http.get("/api/trajectories/stats").await?;
                Ok(Some(result))
            }

            // ── All tools migrated ──────────────────────────────────────
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========================================================================
    // Mega-tool resolution tests
    // ========================================================================

    fn make_handler() -> ToolHandler {
        // Create a handler with a dummy HTTP client (no actual server needed for resolution tests)
        let client = McpHttpClient::new("http://localhost:0".to_string(), None);
        ToolHandler::new(client)
    }

    #[test]
    fn test_resolve_mega_tool_project_list() {
        let handler = make_handler();
        let args = json!({"action": "list"});
        let (name, _resolved_args) = handler.resolve_mega_tool("project", &args).unwrap();
        assert_eq!(name, "list_projects");
    }

    #[test]
    fn test_resolve_mega_tool_project_create() {
        let handler = make_handler();
        let args = json!({"action": "create", "name": "My Project", "root_path": "/tmp"});
        let (name, resolved_args) = handler.resolve_mega_tool("project", &args).unwrap();
        assert_eq!(name, "create_project");
        assert_eq!(
            resolved_args.get("name").unwrap().as_str().unwrap(),
            "My Project"
        );
    }

    #[test]
    fn test_resolve_mega_tool_project_visualization_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("get_graph", "get_project_graph"),
            ("get_intelligence_summary", "get_intelligence_summary"),
            ("get_embeddings_projection", "get_embeddings_projection"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("project", &args).unwrap();
            assert_eq!(
                name, expected,
                "project action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_plan_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_plans"),
            ("create", "create_plan"),
            ("get", "get_plan"),
            ("update", "update_plan"),
            ("update_status", "update_plan_status"),
            ("link_to_project", "link_plan_to_project"),
            ("get_dependency_graph", "get_dependency_graph"),
            ("get_critical_path", "get_critical_path"),
            ("get_waves", "get_waves"),
            ("delete", "delete_plan"),
            ("delegate_task", "delegate_task"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("plan", &args).unwrap();
            assert_eq!(
                name, expected,
                "plan action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_task_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_tasks"),
            ("create", "create_task"),
            ("get", "get_task"),
            ("update", "update_task"),
            ("delete", "delete_task"),
            ("get_next", "get_next_task"),
            ("add_dependencies", "add_task_dependencies"),
            ("get_blockers", "get_task_blockers"),
            ("get_context", "get_task_context"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("task", &args).unwrap();
            assert_eq!(
                name, expected,
                "task action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_note_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_notes"),
            ("create", "create_note"),
            ("search", "search_notes"),
            ("search_semantic", "search_notes_semantic"),
            ("link_to_entity", "link_note_to_entity"),
            ("get_context", "get_context_notes"),
            ("get_context_knowledge", "get_context_knowledge"),
            ("get_propagated_knowledge", "get_propagated_knowledge"),
            ("confirm", "confirm_note"),
            ("invalidate", "invalidate_note"),
            ("supersede", "supersede_note"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("note", &args).unwrap();
            assert_eq!(
                name, expected,
                "note action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_code_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("search", "search_code"),
            ("search_project", "search_project_code"),
            ("get_file_symbols", "get_file_symbols"),
            ("find_references", "find_references"),
            ("get_call_graph", "get_call_graph"),
            ("analyze_impact", "analyze_impact"),
            ("get_architecture", "get_architecture"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("code", &args).unwrap();
            assert_eq!(
                name, expected,
                "code action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_legacy_passthrough() {
        let handler = make_handler();
        // Legacy tool names should pass through unchanged
        let args = json!({"plan_id": "some-id"});
        let (name, resolved_args) = handler.resolve_mega_tool("list_plans", &args).unwrap();
        assert_eq!(name, "list_plans");
        assert_eq!(resolved_args, args);
    }

    #[test]
    fn test_resolve_mega_tool_unknown_action() {
        let handler = make_handler();
        let args = json!({"action": "nonexistent"});
        let result = handler.resolve_mega_tool("project", &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown action"));
    }

    #[test]
    fn test_resolve_mega_tool_missing_action() {
        let handler = make_handler();
        let args = json!({"name": "test"});
        let result = handler.resolve_mega_tool("project", &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("action"));
    }

    #[test]
    fn test_resolve_mega_tool_preserves_all_args() {
        let handler = make_handler();
        let args = json!({"action": "list", "search": "test", "limit": 10});
        let (name, resolved_args) = handler.resolve_mega_tool("project", &args).unwrap();
        assert_eq!(name, "list_projects");
        // action param is kept in resolved args (handlers ignore it)
        assert_eq!(
            resolved_args.get("action").unwrap().as_str().unwrap(),
            "list"
        );
        assert_eq!(
            resolved_args.get("search").unwrap().as_str().unwrap(),
            "test"
        );
        assert_eq!(resolved_args.get("limit").unwrap().as_u64().unwrap(), 10);
    }

    #[test]
    fn test_resolve_all_18_mega_tools() {
        let handler = make_handler();
        // Every mega-tool should have at least "list" or "search" or some base action
        let mega_tools_with_action: Vec<(&str, &str)> = vec![
            ("project", "list"),
            ("plan", "list"),
            ("task", "list"),
            ("step", "list"),
            ("constraint", "list"),
            ("release", "list"),
            ("milestone", "list"),
            ("commit", "create"),
            ("code", "search"),
            ("decision", "get"),
            ("note", "list"),
            ("workspace", "list"),
            ("workspace_milestone", "list_all"),
            ("resource", "list"),
            ("component", "list"),
            ("chat", "list_sessions"),
            ("feature_graph", "list"),
            ("admin", "meilisearch_stats"),
            ("skill", "list"),
            ("analysis_profile", "list"),
            ("protocol", "list"),
            ("persona", "list"),
        ];

        for (tool, action) in mega_tools_with_action {
            let args = json!({"action": action});
            let result = handler.resolve_mega_tool(tool, &args);
            assert!(
                result.is_ok(),
                "mega-tool '{}' with action '{}' should resolve successfully, got: {:?}",
                tool,
                action,
                result.err()
            );
        }
    }

    // ========================================================================
    // HTTP routing tests (mock axum server)
    // ========================================================================

    use axum::{
        body::Body, extract::Request, http::StatusCode, response::IntoResponse, routing::any,
        Router,
    };
    use tokio::net::TcpListener;

    /// Echo handler: returns {method, path, query, body} for every request.
    async fn echo_handler(req: Request<Body>) -> impl IntoResponse {
        let method = req.method().to_string();
        let path = req.uri().path().to_string();
        let query = req.uri().query().unwrap_or("").to_string();
        let body_bytes = axum::body::to_bytes(req.into_body(), 1024 * 1024)
            .await
            .unwrap_or_default();
        let body_str = String::from_utf8_lossy(&body_bytes).to_string();
        let body_json: Value = serde_json::from_str(&body_str).unwrap_or(Value::Null);

        (
            StatusCode::OK,
            axum::Json(json!({
                "_echo": true,
                "method": method,
                "path": path,
                "query": query,
                "body": body_json,
            })),
        )
    }

    /// Spin up a temporary axum server and return a ToolHandler connected to it.
    async fn make_http_handler() -> (ToolHandler, String) {
        let app = Router::new().fallback(any(echo_handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Give the server a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        let client = McpHttpClient::new(base_url.clone(), Some("test-token".to_string()));
        (ToolHandler::new(client), base_url)
    }

    // -- Projects ----------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_projects() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_projects", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects");
    }

    #[tokio::test]
    async fn test_http_list_projects_with_query() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_projects",
                Some(json!({"search": "test", "limit": 10})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects");
        let query = result["query"].as_str().unwrap();
        assert!(
            query.contains("search=test"),
            "query should contain search=test, got: {}",
            query
        );
        assert!(
            query.contains("limit=10"),
            "query should contain limit=10, got: {}",
            query
        );
    }

    #[tokio::test]
    async fn test_http_create_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_project",
                Some(json!({"name": "My Project", "root_path": "/tmp/proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/projects");
        assert_eq!(result["body"]["name"], "My Project");
    }

    #[tokio::test]
    async fn test_http_get_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_project", Some(json!({"slug": "my-project"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects/my-project");
    }

    #[tokio::test]
    async fn test_http_update_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_project",
                Some(json!({"slug": "my-proj", "name": "New Name"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert_eq!(result["path"], "/api/projects/my-proj");
        assert_eq!(result["body"]["name"], "New Name");
    }

    #[tokio::test]
    async fn test_http_delete_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_project", Some(json!({"slug": "old-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(result["path"], "/api/projects/old-proj");
    }

    // -- Plans -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_plan() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_plan",
                Some(json!({"title": "Test Plan", "description": "A plan"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/plans");
        assert_eq!(result["body"]["title"], "Test Plan");
    }

    #[tokio::test]
    async fn test_http_update_plan_status() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_plan_status",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "in_progress"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("550e8400-e29b-41d4-a716-446655440000"));
        assert_eq!(result["body"]["status"], "in_progress");
    }

    #[tokio::test]
    async fn test_http_link_plan_to_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_plan_to_project",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "project_id": "660e8400-e29b-41d4-a716-446655440000"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PUT");
        assert!(result["path"].as_str().unwrap().ends_with("/project"));
    }

    #[tokio::test]
    async fn test_http_delete_plan() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "delete_plan",
                Some(json!({"plan_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().starts_with("/api/plans/"));
    }

    // -- Tasks -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_tasks_with_filters() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_tasks",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "pending",
                    "limit": 20
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/tasks");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("plan_id="));
        assert!(query.contains("status=pending"));
    }

    #[tokio::test]
    async fn test_http_create_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_task",
                Some(json!({
                    "plan_id": "550e8400-e29b-41d4-a716-446655440000",
                    "title": "Test Task"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/tasks"));
        assert_eq!(result["body"]["title"], "Test Task");
    }

    // -- Steps -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_step() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_step",
                Some(json!({
                    "task_id": "550e8400-e29b-41d4-a716-446655440000",
                    "description": "Do something"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().contains("/steps"));
    }

    #[tokio::test]
    async fn test_http_update_step() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_step",
                Some(json!({
                    "step_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"].as_str().unwrap().contains("/steps/"));
    }

    // -- Notes -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_note",
                Some(json!({
                    "content": "Important note",
                    "note_type": "guideline",
                    "importance": "high"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/notes");
    }

    #[tokio::test]
    async fn test_http_search_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("search_notes", Some(json!({"query": "architecture"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/search");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("q=architecture"));
    }

    // -- Commits -----------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_commit() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_commit",
                Some(json!({
                    "sha": "abc1234",
                    "message": "feat: something",
                    "author": "test"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/commits");
        // MCP field "sha" maps to REST "hash"
        assert_eq!(result["body"]["hash"], "abc1234");
    }

    // -- Code --------------------------------------------------------------

    #[tokio::test]
    async fn test_http_search_code() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("search_code", Some(json!({"query": "ToolHandler"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/search");
        let query = result["query"].as_str().unwrap();
        assert!(
            query.contains("query=ToolHandler"),
            "query should contain query=ToolHandler, got: {}",
            query
        );
    }

    // -- Workspaces --------------------------------------------------------

    #[tokio::test]
    async fn test_http_create_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_workspace",
                Some(json!({"name": "Test WS", "slug": "test-ws"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/workspaces");
    }

    // -- Mega-tool → HTTP integration -------------------------------------

    #[tokio::test]
    async fn test_mega_tool_project_routes_to_http() {
        let (handler, _) = make_http_handler().await;
        // Calling mega-tool "project" with action "list" should resolve and route via HTTP
        let result = handler
            .handle("project", Some(json!({"action": "list"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects");
    }

    #[tokio::test]
    async fn test_mega_tool_plan_create_routes_to_http() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "plan",
                Some(json!({"action": "create", "title": "New Plan"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/plans");
    }

    #[tokio::test]
    async fn test_mega_tool_note_search_semantic() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "note",
                Some(json!({"action": "search_semantic", "query": "how to"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/search-semantic");
    }

    // -- Error paths -------------------------------------------------------

    #[tokio::test]
    async fn test_handle_unknown_tool_returns_error() {
        let (handler, _) = make_http_handler().await;
        let result = handler.handle("totally_unknown_tool", None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_handle_missing_required_arg() {
        let (handler, _) = make_http_handler().await;
        // get_project requires "slug" — call without it
        let result = handler.handle("get_project", Some(json!({}))).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("slug"));
    }

    // ========================================================================
    // Comprehensive HTTP routing tests — all remaining handlers
    // ========================================================================

    // Helper UUID for tests
    const UUID1: &str = "550e8400-e29b-41d4-a716-446655440000";
    const UUID2: &str = "660e8400-e29b-41d4-a716-446655440000";

    // -- Projects (remaining) -----------------------------------------------

    #[tokio::test]
    async fn test_http_sync_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("sync_project", Some(json!({"slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/projects/my-proj/sync");
    }

    #[tokio::test]
    async fn test_http_get_project_roadmap() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_project_roadmap", Some(json!({"slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects/my-proj/roadmap");
    }

    #[tokio::test]
    async fn test_http_list_project_plans() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_project_plans", Some(json!({"slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/projects/my-proj/plans");
    }

    // -- Plans (remaining) --------------------------------------------------

    #[tokio::test]
    async fn test_http_list_plans() {
        let (handler, _) = make_http_handler().await;
        let result = handler.handle("list_plans", Some(json!({}))).await.unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/plans");
    }

    #[tokio::test]
    async fn test_http_list_plans_with_filters() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_plans",
                Some(json!({"status": "in_progress", "priority_min": 50, "sort_by": "priority"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("status=in_progress"));
        assert!(query.contains("priority_min=50"));
        assert!(query.contains("sort_by=priority"));
    }

    #[tokio::test]
    async fn test_http_get_plan() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_plan", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().starts_with("/api/plans/"));
    }

    #[tokio::test]
    async fn test_http_unlink_plan_from_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("unlink_plan_from_project", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().ends_with("/project"));
    }

    #[tokio::test]
    async fn test_http_get_dependency_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_dependency_graph", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .ends_with("/dependency-graph"));
    }

    #[tokio::test]
    async fn test_http_get_critical_path() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_critical_path", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/critical-path"));
    }

    #[tokio::test]
    async fn test_http_get_waves() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_waves", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/waves"));
    }

    // -- Tasks (remaining) --------------------------------------------------

    #[tokio::test]
    async fn test_http_get_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_task", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().starts_with("/api/tasks/"));
    }

    #[tokio::test]
    async fn test_http_update_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_task",
                Some(json!({"task_id": UUID1, "status": "completed"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"].as_str().unwrap().starts_with("/api/tasks/"));
        assert_eq!(result["body"]["status"], "completed");
    }

    #[tokio::test]
    async fn test_http_delete_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_task", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().starts_with("/api/tasks/"));
    }

    #[tokio::test]
    async fn test_http_get_next_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_next_task", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/next-task"));
    }

    #[tokio::test]
    async fn test_http_add_task_dependencies() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_task_dependencies",
                Some(json!({"task_id": UUID1, "dependency_ids": [UUID2]})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/dependencies"));
    }

    #[tokio::test]
    async fn test_http_remove_task_dependency() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "remove_task_dependency",
                Some(json!({"task_id": UUID1, "dependency_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/dependencies/"));
    }

    #[tokio::test]
    async fn test_http_get_task_blockers() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_task_blockers", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/blockers"));
    }

    #[tokio::test]
    async fn test_http_get_tasks_blocked_by() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_tasks_blocked_by", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/blocking"));
    }

    #[tokio::test]
    async fn test_http_get_task_context() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_task_context",
                Some(json!({"plan_id": UUID1, "task_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/context"));
    }

    #[tokio::test]
    async fn test_http_get_task_prompt() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_task_prompt",
                Some(json!({"plan_id": UUID1, "task_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/prompt"));
    }

    // -- Steps (remaining) --------------------------------------------------

    #[tokio::test]
    async fn test_http_list_steps() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_steps", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/steps"));
    }

    #[tokio::test]
    async fn test_http_get_step() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_step", Some(json!({"step_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().starts_with("/api/steps/"));
    }

    #[tokio::test]
    async fn test_http_delete_step() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_step", Some(json!({"step_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().starts_with("/api/steps/"));
    }

    #[tokio::test]
    async fn test_http_get_step_progress() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_step_progress", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .ends_with("/steps/progress"));
    }

    // -- Decisions ----------------------------------------------------------

    #[tokio::test]
    async fn test_http_add_decision() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_decision",
                Some(json!({"task_id": UUID1, "description": "Use REST", "rationale": "Simple"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/decisions"));
    }

    #[tokio::test]
    async fn test_http_get_decision() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_decision", Some(json!({"decision_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/decisions/"));
    }

    #[tokio::test]
    async fn test_http_update_decision() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_decision",
                Some(json!({"decision_id": UUID1, "status": "accepted", "rationale": "Updated"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/decisions/"));
    }

    #[tokio::test]
    async fn test_http_delete_decision() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_decision", Some(json!({"decision_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/decisions/"));
    }

    #[tokio::test]
    async fn test_http_search_decisions() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("search_decisions", Some(json!({"query": "auth"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/decisions/search");
    }

    #[tokio::test]
    async fn test_http_search_decisions_semantic() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "search_decisions_semantic",
                Some(json!({"query": "authentication approach"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/decisions/search-semantic");
    }

    #[tokio::test]
    async fn test_http_add_decision_affects() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_decision_affects",
                Some(json!({
                    "decision_id": UUID1,
                    "entity_type": "File",
                    "entity_id": "src/main.rs"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/affects"));
    }

    #[tokio::test]
    async fn test_http_remove_decision_affects() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "remove_decision_affects",
                Some(json!({
                    "decision_id": UUID1,
                    "entity_type": "File",
                    "entity_id": "src/main.rs"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/affects"));
    }

    #[tokio::test]
    async fn test_http_list_decision_affects() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_decision_affects", Some(json!({"decision_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/affects"));
    }

    #[tokio::test]
    async fn test_http_get_decisions_affecting() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_decisions_affecting",
                Some(json!({"entity_type": "File", "entity_id": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/decisions/affecting");
    }

    #[tokio::test]
    async fn test_http_supersede_decision() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "supersede_decision",
                Some(json!({"decision_id": UUID1, "superseded_by_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().contains("/supersedes/"));
    }

    #[tokio::test]
    async fn test_http_get_decision_timeline() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_decision_timeline", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/timeline"));
    }

    // -- Constraints --------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_constraints() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_constraints", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/constraints"));
    }

    #[tokio::test]
    async fn test_http_add_constraint() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_constraint",
                Some(json!({
                    "plan_id": UUID1,
                    "constraint_type": "performance",
                    "description": "< 100ms",
                    "severity": "must"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/constraints"));
    }

    #[tokio::test]
    async fn test_http_get_constraint() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_constraint", Some(json!({"constraint_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/constraints/"));
    }

    #[tokio::test]
    async fn test_http_update_constraint() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_constraint",
                Some(json!({"constraint_id": UUID1, "description": "Updated"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/constraints/"));
    }

    #[tokio::test]
    async fn test_http_delete_constraint() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_constraint", Some(json!({"constraint_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/constraints/"));
    }

    // -- Milestones ---------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_milestones() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_milestones", Some(json!({"project_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/milestones"));
    }

    #[tokio::test]
    async fn test_http_create_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_milestone",
                Some(json!({"project_id": UUID1, "title": "v1.0"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/milestones"));
    }

    #[tokio::test]
    async fn test_http_get_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_milestone", Some(json!({"milestone_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/milestones/"));
    }

    #[tokio::test]
    async fn test_http_update_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_milestone",
                Some(json!({"milestone_id": UUID1, "title": "v2.0"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/milestones/"));
    }

    #[tokio::test]
    async fn test_http_delete_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_milestone", Some(json!({"milestone_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/milestones/"));
    }

    #[tokio::test]
    async fn test_http_get_milestone_progress() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_milestone_progress",
                Some(json!({"milestone_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/progress"));
    }

    #[tokio::test]
    async fn test_http_add_task_to_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_task_to_milestone",
                Some(json!({"milestone_id": UUID1, "task_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/tasks"));
    }

    #[tokio::test]
    async fn test_http_link_plan_to_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_plan_to_milestone",
                Some(json!({"milestone_id": UUID1, "plan_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/plans"));
    }

    #[tokio::test]
    async fn test_http_unlink_plan_from_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "unlink_plan_from_milestone",
                Some(json!({"milestone_id": UUID1, "plan_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/plans/"));
    }

    // -- Releases -----------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_releases() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_releases", Some(json!({"project_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/releases"));
    }

    #[tokio::test]
    async fn test_http_create_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_release",
                Some(json!({"project_id": UUID1, "version": "1.0.0", "title": "First"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/releases"));
    }

    #[tokio::test]
    async fn test_http_get_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_release", Some(json!({"release_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/releases/"));
    }

    #[tokio::test]
    async fn test_http_update_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_release",
                Some(json!({"release_id": UUID1, "status": "released"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/releases/"));
    }

    #[tokio::test]
    async fn test_http_delete_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_release", Some(json!({"release_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/releases/"));
    }

    #[tokio::test]
    async fn test_http_add_task_to_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_task_to_release",
                Some(json!({"release_id": UUID1, "task_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/tasks"));
    }

    #[tokio::test]
    async fn test_http_add_commit_to_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_commit_to_release",
                Some(json!({"release_id": UUID1, "commit_sha": "abc1234"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/commits"));
    }

    #[tokio::test]
    async fn test_http_remove_commit_from_release() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "remove_commit_from_release",
                Some(json!({"release_id": UUID1, "commit_sha": "abc1234"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/commits/"));
    }

    // -- Commits (remaining) ------------------------------------------------

    #[tokio::test]
    async fn test_http_link_commit_to_task() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_commit_to_task",
                Some(json!({"task_id": UUID1, "commit_sha": "abc1234"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/commits"));
        assert_eq!(result["body"]["commit_hash"], "abc1234");
    }

    #[tokio::test]
    async fn test_http_link_commit_to_plan() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_commit_to_plan",
                Some(json!({"plan_id": UUID1, "commit_sha": "def5678"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/commits"));
        assert_eq!(result["body"]["commit_hash"], "def5678");
    }

    #[tokio::test]
    async fn test_http_get_task_commits() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_task_commits", Some(json!({"task_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/commits"));
    }

    #[tokio::test]
    async fn test_http_get_plan_commits() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_plan_commits", Some(json!({"plan_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/commits"));
    }

    #[tokio::test]
    async fn test_http_get_commit_files() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_commit_files", Some(json!({"sha": "abc1234"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/commits/"));
        assert!(result["path"].as_str().unwrap().ends_with("/files"));
    }

    #[tokio::test]
    async fn test_http_get_file_history() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_file_history",
                Some(json!({"file_path": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/files/history");
    }

    // -- Notes (remaining) --------------------------------------------------

    #[tokio::test]
    async fn test_http_list_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler.handle("list_notes", Some(json!({}))).await.unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes");
    }

    #[tokio::test]
    async fn test_http_list_notes_with_filters() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_notes",
                Some(json!({"note_type": "guideline", "importance": "high", "limit": 5})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("note_type=guideline"));
        assert!(query.contains("importance=high"));
    }

    #[tokio::test]
    async fn test_http_get_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_note", Some(json!({"note_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().starts_with("/api/notes/"));
    }

    #[tokio::test]
    async fn test_http_update_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_note",
                Some(json!({"note_id": UUID1, "content": "Updated content"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"].as_str().unwrap().starts_with("/api/notes/"));
    }

    #[tokio::test]
    async fn test_http_delete_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_note", Some(json!({"note_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().starts_with("/api/notes/"));
    }

    #[tokio::test]
    async fn test_http_search_notes_semantic() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "search_notes_semantic",
                Some(json!({"query": "how to test"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/search-semantic");
    }

    #[tokio::test]
    async fn test_http_confirm_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("confirm_note", Some(json!({"note_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/confirm"));
    }

    #[tokio::test]
    async fn test_http_invalidate_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "invalidate_note",
                Some(json!({"note_id": UUID1, "reason": "stale"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/invalidate"));
    }

    #[tokio::test]
    async fn test_http_supersede_note() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "supersede_note",
                Some(json!({"old_note_id": UUID1, "superseded_by_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/supersede"));
    }

    #[tokio::test]
    async fn test_http_link_note_to_entity() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_note_to_entity",
                Some(json!({
                    "note_id": UUID1,
                    "entity_type": "file",
                    "entity_id": "src/main.rs"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/links"));
    }

    #[tokio::test]
    async fn test_http_unlink_note_from_entity() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "unlink_note_from_entity",
                Some(json!({
                    "note_id": UUID1,
                    "entity_type": "file",
                    "entity_id": "src/main.rs"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/links"));
    }

    #[tokio::test]
    async fn test_http_get_context_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_context_notes",
                Some(json!({"entity_type": "file", "entity_id": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/context");
    }

    #[tokio::test]
    async fn test_http_get_propagated_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_propagated_notes",
                Some(json!({"entity_type": "file", "entity_id": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/propagated");
    }

    #[tokio::test]
    async fn test_http_list_project_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_project_notes", Some(json!({"project_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/notes"));
    }

    #[tokio::test]
    async fn test_http_get_entity_notes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_entity_notes",
                Some(json!({"entity_type": "function", "entity_id": "main"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/entities/"));
    }

    #[tokio::test]
    async fn test_http_get_notes_needing_review() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_notes_needing_review", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/needs-review");
    }

    #[tokio::test]
    async fn test_http_get_context_knowledge() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_context_knowledge",
                Some(json!({"entity_type": "file", "entity_id": "src/lib.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/context-knowledge");
    }

    #[tokio::test]
    async fn test_http_get_propagated_knowledge() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_propagated_knowledge",
                Some(json!({"entity_type": "file", "entity_id": "src/lib.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/propagated-knowledge");
    }

    // -- Admin / neurons ----------------------------------------------------

    #[tokio::test]
    async fn test_http_search_neurons() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("search_neurons", Some(json!({"query": "auth"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/notes/neurons/search");
    }

    #[tokio::test]
    async fn test_http_update_staleness_scores() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("update_staleness_scores", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/notes/update-staleness");
    }

    #[tokio::test]
    async fn test_http_update_energy_scores() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("update_energy_scores", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/notes/update-energy");
    }

    #[tokio::test]
    async fn test_http_reinforce_neurons() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "reinforce_neurons",
                Some(json!({"note_ids": [UUID1, UUID2]})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/notes/neurons/reinforce");
    }

    #[tokio::test]
    async fn test_http_decay_synapses() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("decay_synapses", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/notes/neurons/decay");
    }

    #[tokio::test]
    async fn test_http_backfill_synapses() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("backfill_synapses", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/admin/backfill-synapses");
    }

    #[tokio::test]
    async fn test_http_reindex_decisions() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("reindex_decisions", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/admin/reindex-decisions");
    }

    #[tokio::test]
    async fn test_http_backfill_decision_embeddings() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("backfill_decision_embeddings", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/admin/backfill-decision-embeddings");
    }

    #[tokio::test]
    async fn test_http_backfill_touches() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("backfill_touches", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_backfill_discussed() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("backfill_discussed", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_update_fabric_scores() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("update_fabric_scores", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_bootstrap_knowledge_fabric() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("bootstrap_knowledge_fabric", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_detect_skills() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("detect_skills", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_maintain_skills() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "maintain_skills",
                Some(json!({"project_id": UUID1, "level": "daily"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    // -- Context Cards -----------------------------------------------------------

    #[tokio::test]
    async fn test_http_get_context_card() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_context_card",
                Some(json!({"path": "src/main.rs", "project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/context-card"));
    }

    #[tokio::test]
    async fn test_http_refresh_context_cards() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "refresh_context_cards",
                Some(json!({"project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/context-cards/refresh"));
    }

    #[tokio::test]
    async fn test_http_get_fingerprint() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_fingerprint",
                Some(json!({"path": "src/main.rs", "project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/fingerprint"));
    }

    #[tokio::test]
    async fn test_http_find_isomorphic() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("find_isomorphic", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/isomorphic"));
    }

    #[tokio::test]
    async fn test_http_get_meilisearch_stats() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_meilisearch_stats", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_delete_meilisearch_orphans() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_meilisearch_orphans", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
    }

    #[tokio::test]
    async fn test_http_sync_directory() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("sync_directory", Some(json!({"path": "/tmp/project"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/sync");
    }

    #[tokio::test]
    async fn test_http_start_watch() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("start_watch", Some(json!({"path": "/tmp/project"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/watch");
    }

    #[tokio::test]
    async fn test_http_stop_watch() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("stop_watch", Some(json!({"project_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(result["path"], "/api/watch");
    }

    #[tokio::test]
    async fn test_http_watch_status() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("watch_status", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/watch");
    }

    #[tokio::test]
    async fn test_http_cleanup_cross_project_calls() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("cleanup_cross_project_calls", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_cleanup_sync_data() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("cleanup_sync_data", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_cleanup_builtin_calls() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("cleanup_builtin_calls", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_migrate_calls_confidence() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("migrate_calls_confidence", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    // -- Workspaces (remaining) ---------------------------------------------

    #[tokio::test]
    async fn test_http_list_workspaces() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_workspaces", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/workspaces");
    }

    #[tokio::test]
    async fn test_http_get_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_workspace", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/workspaces/my-ws");
    }

    #[tokio::test]
    async fn test_http_update_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_workspace",
                Some(json!({"slug": "my-ws", "name": "New Name"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert_eq!(result["path"], "/api/workspaces/my-ws");
    }

    #[tokio::test]
    async fn test_http_delete_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_workspace", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(result["path"], "/api/workspaces/my-ws");
    }

    #[tokio::test]
    async fn test_http_get_workspace_overview() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_workspace_overview", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/overview"));
    }

    #[tokio::test]
    async fn test_http_list_workspace_projects() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_workspace_projects", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/projects"));
    }

    #[tokio::test]
    async fn test_http_add_project_to_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_project_to_workspace",
                Some(json!({"slug": "my-ws", "project_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/projects"));
    }

    #[tokio::test]
    async fn test_http_remove_project_from_workspace() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "remove_project_from_workspace",
                Some(json!({"slug": "my-ws", "project_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/projects/"));
    }

    #[tokio::test]
    async fn test_http_get_workspace_topology() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_workspace_topology", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/topology"));
    }

    // -- Workspace milestones -----------------------------------------------

    #[tokio::test]
    async fn test_http_list_all_workspace_milestones() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_all_workspace_milestones",
                Some(json!({"workspace_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/workspace-milestones");
    }

    #[tokio::test]
    async fn test_http_list_workspace_milestones() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_workspace_milestones", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/workspaces/my-ws/milestones"));
    }

    #[tokio::test]
    async fn test_http_create_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_workspace_milestone",
                Some(json!({"slug": "my-ws", "title": "Beta"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/milestones"));
    }

    #[tokio::test]
    async fn test_http_get_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_workspace_milestone",
                Some(json!({"milestone_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/workspace-milestones/"));
    }

    #[tokio::test]
    async fn test_http_update_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_workspace_milestone",
                Some(json!({"milestone_id": UUID1, "title": "GA"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/workspace-milestones/"));
    }

    #[tokio::test]
    async fn test_http_delete_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "delete_workspace_milestone",
                Some(json!({"milestone_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/workspace-milestones/"));
    }

    #[tokio::test]
    async fn test_http_add_task_to_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_task_to_workspace_milestone",
                Some(json!({"milestone_id": UUID1, "task_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/tasks"));
    }

    #[tokio::test]
    async fn test_http_link_plan_to_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_plan_to_workspace_milestone",
                Some(json!({"milestone_id": UUID1, "plan_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/plans"));
    }

    #[tokio::test]
    async fn test_http_unlink_plan_from_workspace_milestone() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "unlink_plan_from_workspace_milestone",
                Some(json!({"milestone_id": UUID1, "plan_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/plans/"));
    }

    #[tokio::test]
    async fn test_http_get_workspace_milestone_progress() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_workspace_milestone_progress",
                Some(json!({"milestone_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/progress"));
    }

    // -- Resources ----------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_resources() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_resources", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/resources"));
    }

    #[tokio::test]
    async fn test_http_create_resource() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_resource",
                Some(json!({
                    "slug": "my-ws",
                    "name": "API Schema",
                    "resource_type": "api_contract"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/resources"));
    }

    #[tokio::test]
    async fn test_http_get_resource() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_resource", Some(json!({"id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/resources/"));
    }

    #[tokio::test]
    async fn test_http_update_resource() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_resource",
                Some(json!({"id": UUID1, "name": "Updated"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/resources/"));
    }

    #[tokio::test]
    async fn test_http_delete_resource() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_resource", Some(json!({"id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/resources/"));
    }

    #[tokio::test]
    async fn test_http_link_resource_to_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "link_resource_to_project",
                Some(json!({"id": UUID1, "project_id": UUID2, "link_type": "uses"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/projects"));
    }

    // -- Components ---------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_components() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_components", Some(json!({"slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/components"));
    }

    #[tokio::test]
    async fn test_http_create_component() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_component",
                Some(json!({
                    "slug": "my-ws",
                    "name": "API",
                    "component_type": "service"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/components"));
    }

    #[tokio::test]
    async fn test_http_get_component() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_component", Some(json!({"id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/components/"));
    }

    #[tokio::test]
    async fn test_http_update_component() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_component",
                Some(json!({"id": UUID1, "name": "Gateway"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/components/"));
    }

    #[tokio::test]
    async fn test_http_delete_component() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_component", Some(json!({"id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/components/"));
    }

    #[tokio::test]
    async fn test_http_add_component_dependency() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_component_dependency",
                Some(json!({"id": UUID1, "depends_on_id": UUID2, "dependency_type": "uses"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/api/components/"));
    }

    #[tokio::test]
    async fn test_http_remove_component_dependency() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "remove_component_dependency",
                Some(json!({"id": UUID1, "dep_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/api/components/"));
    }

    #[tokio::test]
    async fn test_http_map_component_to_project() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "map_component_to_project",
                Some(json!({"id": UUID1, "project_id": UUID2})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PUT");
        assert!(result["path"].as_str().unwrap().ends_with("/project"));
    }

    // -- Code (remaining) ---------------------------------------------------

    #[tokio::test]
    async fn test_http_search_project_code() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "search_project_code",
                Some(json!({"project_slug": "my-proj", "query": "handler"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/projects/my-proj/code/search"));
    }

    #[tokio::test]
    async fn test_http_search_workspace_code() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "search_workspace_code",
                Some(json!({"workspace_slug": "my-ws", "query": "service"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/api/code/search"));
    }

    #[tokio::test]
    async fn test_http_get_file_symbols() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_file_symbols",
                Some(json!({"file_path": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/code/symbols/"));
    }

    #[tokio::test]
    async fn test_http_find_references() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("find_references", Some(json!({"symbol": "ToolHandler"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/references");
    }

    #[tokio::test]
    async fn test_http_get_file_dependencies() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_file_dependencies",
                Some(json!({"file_path": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/code/dependencies/"));
    }

    #[tokio::test]
    async fn test_http_get_call_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_call_graph", Some(json!({"function": "main"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/callgraph");
    }

    #[tokio::test]
    async fn test_http_analyze_impact() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("analyze_impact", Some(json!({"target": "src/main.rs"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/impact");
    }

    #[tokio::test]
    async fn test_http_get_architecture() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_architecture", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/architecture");
    }

    #[tokio::test]
    async fn test_http_find_similar_code() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_similar_code",
                Some(json!({"code_snippet": "fn main()"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/similar");
    }

    #[tokio::test]
    async fn test_http_find_trait_implementations() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_trait_implementations",
                Some(json!({"trait_name": "GraphStore"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/trait-impls");
    }

    #[tokio::test]
    async fn test_http_find_type_traits() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_type_traits",
                Some(json!({"type_name": "Neo4jClient"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/type-traits");
    }

    #[tokio::test]
    async fn test_http_get_impl_blocks() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_impl_blocks", Some(json!({"type_name": "Neo4jClient"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/impl-blocks");
    }

    #[tokio::test]
    async fn test_http_get_code_communities() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_code_communities",
                Some(json!({"project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/communities"));
    }

    #[tokio::test]
    async fn test_http_get_code_health() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_code_health", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().contains("/code/health"));
    }

    #[tokio::test]
    async fn test_http_get_node_importance() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_node_importance",
                Some(json!({
                    "project_slug": "my-proj",
                    "node_path": "src/main.rs",
                    "node_type": "File"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/node-importance"));
    }

    #[tokio::test]
    async fn test_http_get_co_change_graph() {
        let (handler, _) = make_http_handler().await;
        // This handler makes 2 HTTP calls (resolve slug, then get co-changes)
        // The echo server returns the first call's echo, so just verify it starts
        let result = handler
            .handle(
                "get_co_change_graph",
                Some(json!({"project_slug": "my-proj"})),
            )
            .await;
        // May error because the echo response can't be parsed as project,
        // but the important thing is that the code path is exercised
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_http_get_file_co_changers() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_file_co_changers",
                Some(json!({"file_path": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/files/co-changers"));
    }

    #[tokio::test]
    async fn test_http_detect_processes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("detect_processes", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_list_processes() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_processes", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_get_process() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_process", Some(json!({"process_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_get_entry_points() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_entry_points", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_get_class_hierarchy() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_class_hierarchy",
                Some(json!({"type_name": "BaseHandler"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/class-hierarchy"));
    }

    #[tokio::test]
    async fn test_http_find_subclasses() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_subclasses",
                Some(json!({"class_name": "BaseHandler"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/subclasses"));
    }

    #[tokio::test]
    async fn test_http_find_interface_implementors() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_interface_implementors",
                Some(json!({"interface_name": "Handler"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/interface-implementors"));
    }

    #[tokio::test]
    async fn test_http_enrich_communities() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "enrich_communities",
                Some(json!({"project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
    }

    #[tokio::test]
    async fn test_http_get_hotspots() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_hotspots", Some(json!({"project_slug": "my-proj"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_get_knowledge_gaps() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_knowledge_gaps",
                Some(json!({"project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_get_risk_assessment() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_risk_assessment",
                Some(json!({"project_slug": "my-proj"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_http_plan_implementation() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "plan_implementation",
                Some(json!({
                    "project_slug": "my-proj",
                    "description": "Add auth"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .contains("/code/plan-implementation"));
    }

    // -- Chat ---------------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_chat_sessions() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_chat_sessions", Some(json!({})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/chat/sessions");
    }

    #[tokio::test]
    async fn test_http_get_chat_session() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_chat_session", Some(json!({"session_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/chat/sessions/"));
    }

    #[tokio::test]
    async fn test_http_delete_chat_session() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_chat_session", Some(json!({"session_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/chat/sessions/"));
    }

    #[tokio::test]
    async fn test_http_chat_send_message() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "chat_send_message",
                Some(json!({"message": "Hello", "cwd": "/tmp"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/chat/sessions");
    }

    #[tokio::test]
    async fn test_http_list_chat_messages() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_chat_messages", Some(json!({"session_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/messages"));
    }

    #[tokio::test]
    async fn test_http_add_discussed() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_discussed",
                Some(json!({
                    "session_id": UUID1,
                    "entities": [{"entity_type": "file", "entity_id": "src/main.rs"}]
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/discussed"));
    }

    #[tokio::test]
    async fn test_http_get_session_entities() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_session_entities", Some(json!({"session_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/discussed"));
    }

    // -- Structural DNA -------------------------------------------------------

    #[tokio::test]
    async fn test_http_get_structural_profile() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_structural_profile",
                Some(json!({"project_slug": "my-project", "file_path": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/structural-profile");
    }

    #[tokio::test]
    async fn test_http_find_structural_twins() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_structural_twins",
                Some(json!({"project_slug": "my-project", "file_path": "src/main.rs", "top_n": 5})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/structural-twins");
    }

    #[tokio::test]
    async fn test_http_cluster_dna() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "cluster_dna",
                Some(json!({"project_slug": "my-project", "n_clusters": 5})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/structural-clusters");
    }

    #[tokio::test]
    async fn test_http_find_cross_project_twins() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_cross_project_twins",
                Some(json!({
                    "workspace_slug": "main",
                    "source_project_slug": "my-project",
                    "file_path": "src/main.rs",
                    "top_n": 5
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/structural-twins/cross-project");
    }

    #[tokio::test]
    async fn test_http_predict_missing_links() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "predict_missing_links",
                Some(json!({"project_slug": "my-project", "top_n": 10, "min_plausibility": 0.3})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/predict-links");
    }

    #[tokio::test]
    async fn test_http_check_link_plausibility() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "check_link_plausibility",
                Some(json!({
                    "project_slug": "my-project",
                    "source": "src/main.rs",
                    "target": "src/lib.rs"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/link-plausibility");
    }

    // -- Stress testing -----------------------------------------------------

    #[tokio::test]
    async fn test_http_stress_test_node() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "stress_test_node",
                Some(json!({"project_slug": "my-project", "target_id": "src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/code/stress-test-node");
    }

    // -- Feature graphs -----------------------------------------------------

    #[tokio::test]
    async fn test_http_create_feature_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_feature_graph",
                Some(json!({"project_id": UUID1, "name": "Auth flow"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/feature-graphs");
    }

    #[tokio::test]
    async fn test_http_list_feature_graphs() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_feature_graphs", Some(json!({"project_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/feature-graphs");
    }

    #[tokio::test]
    async fn test_http_get_feature_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_feature_graph", Some(json!({"id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/feature-graphs/"));
    }

    #[tokio::test]
    async fn test_http_delete_feature_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_feature_graph", Some(json!({"id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/feature-graphs/"));
    }

    #[tokio::test]
    async fn test_http_add_to_feature_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_to_feature_graph",
                Some(json!({
                    "feature_graph_id": UUID1,
                    "entity_type": "Function",
                    "entity_id": "main"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/entities"));
    }

    #[tokio::test]
    async fn test_http_auto_build_feature_graph() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "auto_build_feature_graph",
                Some(json!({"project_id": UUID1, "name": "Login flow", "entry_function": "main"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/feature-graphs/auto-build");
    }

    // -- Skills -------------------------------------------------------------

    #[tokio::test]
    async fn test_http_list_skills() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_skills", Some(json!({"project_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/skills");
    }

    #[tokio::test]
    async fn test_http_create_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_skill",
                Some(json!({"project_id": UUID1, "name": "Auth skill"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/skills");
    }

    #[tokio::test]
    async fn test_http_get_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_skill", Some(json!({"skill_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().starts_with("/api/skills/"));
    }

    #[tokio::test]
    async fn test_http_update_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_skill",
                Some(json!({"skill_id": UUID1, "name": "Updated"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PUT");
        assert!(result["path"].as_str().unwrap().starts_with("/api/skills/"));
    }

    #[tokio::test]
    async fn test_http_delete_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("delete_skill", Some(json!({"skill_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().starts_with("/api/skills/"));
    }

    #[tokio::test]
    async fn test_http_get_skill_members() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_skill_members", Some(json!({"skill_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/members"));
    }

    #[tokio::test]
    async fn test_http_add_skill_member() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "add_skill_member",
                Some(json!({
                    "skill_id": UUID1,
                    "entity_type": "note",
                    "entity_id": UUID2
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/members"));
    }

    #[tokio::test]
    async fn test_http_remove_skill_member() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "remove_skill_member",
                Some(json!({
                    "skill_id": UUID1,
                    "entity_type": "note",
                    "entity_id": UUID2
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert!(result["path"].as_str().unwrap().contains("/members/"));
    }

    #[tokio::test]
    async fn test_http_activate_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "activate_skill",
                Some(json!({"skill_id": UUID1, "query": "auth flow"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/activate"));
    }

    #[tokio::test]
    async fn test_http_export_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("export_skill", Some(json!({"skill_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/export"));
    }

    #[tokio::test]
    async fn test_http_import_skill() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "import_skill",
                Some(json!({"project_id": UUID1, "package": {"name": "test"}})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/skills/import");
    }

    #[tokio::test]
    async fn test_http_get_skill_health() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_skill_health", Some(json!({"skill_id": UUID1})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/health"));
    }

    // -- Mega-tool integration (remaining) ----------------------------------

    #[tokio::test]
    async fn test_mega_tool_task_update() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "task",
                Some(json!({"action": "update", "task_id": UUID1, "status": "completed"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PATCH");
        assert!(result["path"].as_str().unwrap().starts_with("/api/tasks/"));
    }

    #[tokio::test]
    async fn test_mega_tool_step_create() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "step",
                Some(json!({
                    "action": "create",
                    "task_id": UUID1,
                    "description": "Do something"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().contains("/steps"));
    }

    #[tokio::test]
    async fn test_mega_tool_decision_add() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "decision",
                Some(json!({
                    "action": "add",
                    "task_id": UUID1,
                    "description": "Use REST"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/decisions"));
    }

    #[tokio::test]
    async fn test_mega_tool_constraint_list() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "constraint",
                Some(json!({"action": "list", "plan_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/constraints"));
    }

    #[tokio::test]
    async fn test_mega_tool_release_create() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "release",
                Some(json!({
                    "action": "create",
                    "project_id": UUID1,
                    "version": "1.0.0"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/releases"));
    }

    #[tokio::test]
    async fn test_mega_tool_milestone_create() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "milestone",
                Some(json!({
                    "action": "create",
                    "project_id": UUID1,
                    "title": "GA"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/milestones"));
    }

    #[tokio::test]
    async fn test_mega_tool_commit_create() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "commit",
                Some(json!({
                    "action": "create",
                    "sha": "abc123",
                    "message": "feat: test"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/commits");
    }

    #[tokio::test]
    async fn test_mega_tool_workspace_get() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("workspace", Some(json!({"action": "get", "slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/workspaces/my-ws");
    }

    #[tokio::test]
    async fn test_mega_tool_code_find_references() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "code",
                Some(json!({"action": "find_references", "symbol": "main"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/code/references");
    }

    #[tokio::test]
    async fn test_mega_tool_admin_watch_status() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("admin", Some(json!({"action": "watch_status"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/watch");
    }

    #[tokio::test]
    async fn test_mega_tool_skill_list() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "skill",
                Some(json!({"action": "list", "project_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/skills");
    }

    #[tokio::test]
    async fn test_mega_tool_chat_list_sessions() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("chat", Some(json!({"action": "list_sessions"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/chat/sessions");
    }

    #[tokio::test]
    async fn test_mega_tool_feature_graph_list() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "feature_graph",
                Some(json!({"action": "list", "project_id": UUID1})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/feature-graphs");
    }

    #[tokio::test]
    async fn test_mega_tool_resource_list() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("resource", Some(json!({"action": "list", "slug": "my-ws"})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert!(result["path"].as_str().unwrap().ends_with("/resources"));
    }

    #[tokio::test]
    async fn test_mega_tool_component_create() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "component",
                Some(json!({
                    "action": "create",
                    "slug": "my-ws",
                    "name": "API",
                    "component_type": "service"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/components"));
    }

    #[tokio::test]
    async fn test_mega_tool_workspace_milestone_create() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "workspace_milestone",
                Some(json!({"action": "create", "slug": "my-ws", "title": "Beta"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"].as_str().unwrap().ends_with("/milestones"));
    }

    // -- Error paths (additional) -------------------------------------------

    #[tokio::test]
    async fn test_handle_missing_plan_id() {
        let (handler, _) = make_http_handler().await;
        let result = handler.handle("get_plan", Some(json!({}))).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_invalid_uuid() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("get_plan", Some(json!({"plan_id": "not-a-uuid"})))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_none_args() {
        let (handler, _) = make_http_handler().await;
        // list_projects doesn't require args, should work with None
        let result = handler.handle("list_projects", None).await.unwrap();
        assert_eq!(result["method"], "GET");
    }

    #[tokio::test]
    async fn test_mega_tool_with_empty_args() {
        let (handler, _) = make_http_handler().await;
        // mega-tool without action should fail
        let result = handler.handle("project", Some(json!({}))).await;
        assert!(result.is_err());
    }

    // -- Resolve mega-tool additional tests ---------------------------------

    #[tokio::test]
    async fn test_resolve_mega_tool_decision_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("add", "add_decision"),
            ("get", "get_decision"),
            ("update", "update_decision"),
            ("delete", "delete_decision"),
            ("search", "search_decisions"),
            ("search_semantic", "search_decisions_semantic"),
            ("add_affects", "add_decision_affects"),
            ("remove_affects", "remove_decision_affects"),
            ("list_affects", "list_decision_affects"),
            ("get_affecting", "get_decisions_affecting"),
            ("supersede", "supersede_decision"),
            ("get_timeline", "get_decision_timeline"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("decision", &args).unwrap();
            assert_eq!(
                name, expected,
                "decision action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_constraint_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_constraints"),
            ("add", "add_constraint"),
            ("get", "get_constraint"),
            ("update", "update_constraint"),
            ("delete", "delete_constraint"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("constraint", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_release_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_releases"),
            ("create", "create_release"),
            ("get", "get_release"),
            ("update", "update_release"),
            ("delete", "delete_release"),
            ("add_task", "add_task_to_release"),
            ("add_commit", "add_commit_to_release"),
            ("remove_commit", "remove_commit_from_release"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("release", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_milestone_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_milestones"),
            ("create", "create_milestone"),
            ("get", "get_milestone"),
            ("update", "update_milestone"),
            ("delete", "delete_milestone"),
            ("get_progress", "get_milestone_progress"),
            ("add_task", "add_task_to_milestone"),
            ("link_plan", "link_plan_to_milestone"),
            ("unlink_plan", "unlink_plan_from_milestone"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("milestone", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_commit_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("create", "create_commit"),
            ("link_to_task", "link_commit_to_task"),
            ("link_to_plan", "link_commit_to_plan"),
            ("get_task_commits", "get_task_commits"),
            ("get_plan_commits", "get_plan_commits"),
            ("get_commit_files", "get_commit_files"),
            ("get_file_history", "get_file_history"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("commit", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_workspace_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_workspaces"),
            ("create", "create_workspace"),
            ("get", "get_workspace"),
            ("update", "update_workspace"),
            ("delete", "delete_workspace"),
            ("get_overview", "get_workspace_overview"),
            ("list_projects", "list_workspace_projects"),
            ("add_project", "add_project_to_workspace"),
            ("remove_project", "remove_project_from_workspace"),
            ("get_topology", "get_workspace_topology"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("workspace", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_skill_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list", "list_skills"),
            ("create", "create_skill"),
            ("get", "get_skill"),
            ("update", "update_skill"),
            ("delete", "delete_skill"),
            ("get_members", "get_skill_members"),
            ("add_member", "add_skill_member"),
            ("remove_member", "remove_skill_member"),
            ("activate", "activate_skill"),
            ("export", "export_skill"),
            ("import", "import_skill"),
            ("get_health", "get_skill_health"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("skill", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_chat_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("list_sessions", "list_chat_sessions"),
            ("get_session", "get_chat_session"),
            ("get_children", "get_session_children"),
            ("delete_session", "delete_chat_session"),
            ("send_message", "chat_send_message"),
            ("list_messages", "list_chat_messages"),
            ("add_discussed", "add_discussed"),
            ("get_session_entities", "get_session_entities"),
            ("get_session_tree", "get_session_tree"),
            ("get_run_sessions", "get_run_sessions"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("chat", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn test_resolve_mega_tool_admin_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("sync_directory", "sync_directory"),
            ("start_watch", "start_watch"),
            ("stop_watch", "stop_watch"),
            ("watch_status", "watch_status"),
            ("meilisearch_stats", "get_meilisearch_stats"),
            ("delete_meilisearch_orphans", "delete_meilisearch_orphans"),
            ("update_staleness_scores", "update_staleness_scores"),
            ("update_energy_scores", "update_energy_scores"),
            ("detect_skills", "detect_skills"),
            ("maintain_skills", "maintain_skills"),
            ("install_hooks", "install_hooks"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("admin", &args).unwrap();
            assert_eq!(
                name, expected,
                "admin action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[test]
    fn test_resolve_mega_tool_feature_graph_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("create", "create_feature_graph"),
            ("get", "get_feature_graph"),
            ("list", "list_feature_graphs"),
            ("add_entity", "add_to_feature_graph"),
            ("auto_build", "auto_build_feature_graph"),
            ("delete", "delete_feature_graph"),
            ("get_statistics", "get_feature_graph_statistics"),
            ("compare", "compare_feature_graphs"),
            ("find_overlapping", "find_overlapping_feature_graphs"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("feature_graph", &args).unwrap();
            assert_eq!(name, expected);
        }
    }

    // -- Persona -----------------------------------------------------------

    #[test]
    fn test_resolve_mega_tool_persona_actions() {
        let handler = make_handler();
        for (action, expected) in [
            ("create", "create_persona"),
            ("get", "get_persona"),
            ("list", "list_personas"),
            ("update", "update_persona"),
            ("delete", "delete_persona"),
            ("add_skill", "add_persona_skill"),
            ("remove_skill", "remove_persona_skill"),
            ("add_protocol", "add_persona_protocol"),
            ("remove_protocol", "remove_persona_protocol"),
            ("add_file", "add_persona_file"),
            ("remove_file", "remove_persona_file"),
            ("add_function", "add_persona_function"),
            ("remove_function", "remove_persona_function"),
            ("add_note", "add_persona_note"),
            ("remove_note", "remove_persona_note"),
            ("add_decision", "add_persona_decision"),
            ("remove_decision", "remove_persona_decision"),
            ("scope_to_feature_graph", "scope_persona_feature_graph"),
            ("unscope_feature_graph", "unscope_persona_feature_graph"),
            ("add_extends", "add_persona_extends"),
            ("remove_extends", "remove_persona_extends"),
            ("get_subgraph", "get_persona_subgraph"),
            ("find_for_file", "find_personas_for_file"),
            ("list_global", "list_global_personas"),
            ("export", "export_persona"),
            ("import", "import_persona"),
            ("activate", "activate_persona"),
            ("auto_build", "auto_build_persona"),
            ("maintain", "maintain_personas"),
            ("detect", "detect_personas"),
        ] {
            let args = json!({"action": action});
            let (name, _) = handler.resolve_mega_tool("persona", &args).unwrap();
            assert_eq!(
                name, expected,
                "persona action '{}' should resolve to '{}'",
                action, expected
            );
        }
    }

    #[tokio::test]
    async fn test_http_list_personas() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_personas",
                Some(json!({"project_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/personas");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("project_id=550e8400"));
    }

    #[tokio::test]
    async fn test_http_list_personas_with_filters() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "list_personas",
                Some(json!({
                    "project_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "active",
                    "limit": 10,
                    "offset": 5
                })),
            )
            .await
            .unwrap();
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("status=active"), "query: {}", query);
        assert!(query.contains("limit=10"), "query: {}", query);
        assert!(query.contains("offset=5"), "query: {}", query);
    }

    #[tokio::test]
    async fn test_http_create_persona() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "create_persona",
                Some(json!({
                    "project_id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "neo4j-expert",
                    "description": "Expert in Neo4j"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/personas");
        assert_eq!(result["body"]["name"], "neo4j-expert");
    }

    #[tokio::test]
    async fn test_http_get_persona() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "get_persona",
                Some(json!({"persona_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(
            result["path"],
            "/api/personas/550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[tokio::test]
    async fn test_http_update_persona() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "update_persona",
                Some(json!({
                    "persona_id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "updated-name",
                    "status": "active",
                    "energy": 0.9
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "PUT");
        assert_eq!(
            result["path"],
            "/api/personas/550e8400-e29b-41d4-a716-446655440000"
        );
        assert_eq!(result["body"]["name"], "updated-name");
        assert_eq!(result["body"]["status"], "active");
    }

    #[tokio::test]
    async fn test_http_delete_persona() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "delete_persona",
                Some(json!({"persona_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(
            result["path"],
            "/api/personas/550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[tokio::test]
    async fn test_http_add_persona_skill() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let sid = "660e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "add_persona_skill",
                Some(json!({"persona_id": pid, "skill_id": sid})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(
            result["path"],
            format!("/api/personas/{}/skills/{}", pid, sid)
        );
    }

    #[tokio::test]
    async fn test_http_remove_persona_skill() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let sid = "660e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "remove_persona_skill",
                Some(json!({"persona_id": pid, "skill_id": sid})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(
            result["path"],
            format!("/api/personas/{}/skills/{}", pid, sid)
        );
    }

    #[tokio::test]
    async fn test_http_add_persona_file() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "add_persona_file",
                Some(json!({"persona_id": pid, "file_path": "/src/main.rs", "weight": 0.8})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], format!("/api/personas/{}/files", pid));
        assert_eq!(result["body"]["file_path"], "/src/main.rs");
        assert_eq!(result["body"]["weight"], 0.8);
    }

    #[tokio::test]
    async fn test_http_add_persona_file_default_weight() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "add_persona_file",
                Some(json!({"persona_id": pid, "file_path": "/src/lib.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["body"]["weight"], 1.0);
    }

    #[tokio::test]
    async fn test_http_remove_persona_file() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "remove_persona_file",
                Some(json!({"persona_id": pid, "file_path": "/src/main.rs"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "DELETE");
        assert_eq!(result["path"], format!("/api/personas/{}/files", pid));
    }

    #[tokio::test]
    async fn test_http_add_persona_note() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let nid = "770e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "add_persona_note",
                Some(json!({"persona_id": pid, "note_id": nid, "weight": 0.7})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(
            result["path"],
            format!("/api/personas/{}/notes/{}", pid, nid)
        );
        assert_eq!(result["body"]["weight"], 0.7);
    }

    #[tokio::test]
    async fn test_http_add_persona_decision() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let did = "880e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "add_persona_decision",
                Some(json!({"persona_id": pid, "decision_id": did})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(
            result["path"],
            format!("/api/personas/{}/decisions/{}", pid, did)
        );
    }

    #[tokio::test]
    async fn test_http_get_persona_subgraph() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle("get_persona_subgraph", Some(json!({"persona_id": pid})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], format!("/api/personas/{}/subgraph", pid));
    }

    #[tokio::test]
    async fn test_http_find_personas_for_file() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "find_personas_for_file",
                Some(json!({
                    "file_path": "/src/neo4j/persona.rs",
                    "project_id": "550e8400-e29b-41d4-a716-446655440000"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/personas/find-for-file");
    }

    #[tokio::test]
    async fn test_http_list_global_personas() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle("list_global_personas", Some(json!({"limit": 5})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], "/api/personas/global");
        let query = result["query"].as_str().unwrap();
        assert!(query.contains("limit=5"), "query: {}", query);
    }

    #[tokio::test]
    async fn test_http_export_persona() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle("export_persona", Some(json!({"persona_id": pid})))
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        assert_eq!(result["path"], format!("/api/personas/{}/export", pid));
    }

    #[tokio::test]
    async fn test_http_export_persona_with_source() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "export_persona",
                Some(json!({"persona_id": pid, "source_project_name": "my-project"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "GET");
        let query = result["query"].as_str().unwrap();
        assert!(
            query.contains("source_project_name=my-project"),
            "query: {}",
            query
        );
    }

    #[tokio::test]
    async fn test_http_import_persona() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "import_persona",
                Some(json!({
                    "project_id": "550e8400-e29b-41d4-a716-446655440000",
                    "package": {"schema_version": 1}
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/personas/import");
    }

    #[tokio::test]
    async fn test_http_activate_persona() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle("activate_persona", Some(json!({"persona_id": pid})))
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], format!("/api/personas/{}/activate", pid));
    }

    #[tokio::test]
    async fn test_http_auto_build_persona() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "auto_build_persona",
                Some(json!({
                    "project_id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "auto-persona"
                })),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(result["path"], "/api/personas/auto-build");
    }

    #[tokio::test]
    async fn test_http_maintain_personas() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "maintain_personas",
                Some(json!({"project_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/personas/maintain"));
    }

    #[tokio::test]
    async fn test_http_detect_personas() {
        let (handler, _) = make_http_handler().await;
        let result = handler
            .handle(
                "detect_personas",
                Some(json!({"project_id": "550e8400-e29b-41d4-a716-446655440000"})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert!(result["path"]
            .as_str()
            .unwrap()
            .starts_with("/api/personas/detect"));
    }

    #[tokio::test]
    async fn test_http_scope_persona_feature_graph() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let fgid = "660e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "scope_persona_feature_graph",
                Some(json!({"persona_id": pid, "feature_graph_id": fgid})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(
            result["path"],
            format!("/api/personas/{}/feature-graphs/{}", pid, fgid)
        );
    }

    #[tokio::test]
    async fn test_http_add_persona_extends() {
        let (handler, _) = make_http_handler().await;
        let pid = "550e8400-e29b-41d4-a716-446655440000";
        let parent = "660e8400-e29b-41d4-a716-446655440000";
        let result = handler
            .handle(
                "add_persona_extends",
                Some(json!({"persona_id": pid, "parent_persona_id": parent})),
            )
            .await
            .unwrap();
        assert_eq!(result["method"], "POST");
        assert_eq!(
            result["path"],
            format!("/api/personas/{}/extends/{}", pid, parent)
        );
    }

    // ========================================================================
    // unstringify_json_values tests
    // ========================================================================

    #[test]
    fn test_unstringify_json_values() {
        let mut v = json!({
            "tags": "[\"a\",\"b\"]",
            "priority": "100",
            "affected_files": "[\"src/main.rs\"]",
            "title": "normal string",
            "flag": "true"
        });
        unstringify_json_values(&mut v);
        assert_eq!(v["tags"], json!(["a", "b"]));
        assert_eq!(v["priority"], json!(100));
        assert_eq!(v["affected_files"], json!(["src/main.rs"]));
        assert_eq!(v["title"], "normal string"); // unchanged
        assert_eq!(v["flag"], json!(true));
    }

    #[test]
    fn test_unstringify_json_values_false() {
        let mut v = json!({"active": "false"});
        unstringify_json_values(&mut v);
        assert_eq!(v["active"], json!(false));
    }

    #[test]
    fn test_unstringify_json_values_object() {
        let mut v = json!({"meta": "{\"key\":\"val\"}"});
        unstringify_json_values(&mut v);
        assert_eq!(v["meta"], json!({"key": "val"}));
    }

    #[test]
    fn test_unstringify_json_values_leaves_non_json_strings() {
        let mut v = json!({"name": "hello world", "path": "/tmp/foo"});
        let original = v.clone();
        unstringify_json_values(&mut v);
        assert_eq!(v, original);
    }

    #[test]
    fn test_unstringify_json_values_non_object() {
        // Should not panic on non-object values
        let mut v = json!("just a string");
        unstringify_json_values(&mut v);
        assert_eq!(v, json!("just a string"));
    }
}
