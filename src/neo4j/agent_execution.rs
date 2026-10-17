//! Neo4j AgentExecution operations — per-agent execution tracking within a PlanRun.
//!
//! Each spawned agent gets its own AgentExecution node, linked to both the PlanRun
//! and the Task it executes. This enables per-agent vector collection and
//! fine-grained historical analysis.

use super::client::Neo4jClient;
use anyhow::Result;
use chrono::{DateTime, Utc};
use neo4rs::query;
use uuid::Uuid;

/// Represents an AgentExecution node in Neo4j.
///
/// Tracks a single agent's execution within a PlanRun, including its cost,
/// status, tools used, files modified, and commits produced.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentExecutionNode {
    pub id: Uuid,
    pub run_id: Uuid,
    pub task_id: Uuid,
    pub session_id: Option<Uuid>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub cost_usd: f64,
    pub duration_secs: f64,
    pub status: AgentExecutionStatus,
    pub tools_used: String,
    pub files_modified: Vec<String>,
    pub commits: Vec<String>,
    pub persona_profile: String,
    /// Per-agent execution vector JSON (serialized AgentExecutionVector)
    pub vector_json: Option<String>,
    /// Structured execution report JSON (serialized TaskExecutionReport)
    pub report_json: Option<String>,
    /// The type of execution (task agent, gate retry, verification).
    #[serde(default)]
    pub execution_type: ExecutionType,

    // ── Per-execution record (A22). All additive: a node written before these
    // fields existed reads back with the defaults below. ──────────────────────
    /// Provider that ran this execution.
    #[serde(default = "default_provider_id")]
    pub provider_id: String,
    /// Model the caller asked for (`None` = provider default).
    #[serde(default)]
    pub model_requested: Option<String>,
    /// Model that actually ran, as reported by the provider.
    #[serde(default)]
    pub model: Option<String>,
    /// Model alias that was requested, when the request went through an alias.
    #[serde(default)]
    pub model_alias: Option<String>,
    /// Resolution level that picked the provider/model (`request`, `task`,
    /// `persona`, `run`, `project_rule`, `global_rule`, `default`, `fallback`…).
    #[serde(default = "default_routed_by")]
    pub routed_by: String,
    /// Policy rule that matched (e.g. `runner.simple`).
    #[serde(default)]
    pub route_rule: Option<String>,
    /// Why a fallback was taken, when one was.
    #[serde(default)]
    pub fallback_reason: Option<String>,
    /// Model the policy would have chosen in shadow mode.
    #[serde(default)]
    pub shadow_model: Option<String>,
    /// Provider the policy would have chosen in shadow mode.
    #[serde(default)]
    pub shadow_provider: Option<String>,
    /// Task class at launch (`simple`, `complex`, `creative`).
    #[serde(default)]
    pub task_class: Option<String>,
    /// Attempt number for this task within the run (1 = first pass).
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    /// `HEAD` of the task working directory when the attempt started.
    #[serde(default)]
    pub base_sha: Option<String>,
    #[serde(default, rename = "input_tokens", alias = "tokens_in")]
    pub tokens_in: Option<u64>,
    #[serde(default, rename = "output_tokens", alias = "tokens_out")]
    pub tokens_out: Option<u64>,
    #[serde(default)]
    pub tokens_cache_read: Option<u64>,
    #[serde(default)]
    pub tokens_cache_write: Option<u64>,
    /// Where `cost_usd` comes from: `reported`, `priced`, `free`,
    /// `subscription` or `unknown`.
    #[serde(default)]
    pub cost_basis: Option<String>,
    /// Number of turns reported by the provider.
    #[serde(default)]
    pub num_turns: Option<u32>,
    /// Per-check verification detail (serialized JSON).
    #[serde(default)]
    pub verification_json: Option<String>,
    /// The verdict of `verification_json`, flat so a Cypher aggregate can read
    /// it (B35). `None` for an attempt nobody verified, and for every node
    /// written before the field existed.
    #[serde(default)]
    pub verification_passed: Option<bool>,
    /// Cognitive routing decision that chose this execution's model (B-R7),
    /// applied or not. `None` for an execution nobody routed, and for every
    /// node written before the field existed.
    #[serde(default)]
    pub routing_decision_id: Option<Uuid>,
}

/// Provider recorded on executions that predate provider selection.
pub const DEFAULT_PROVIDER_ID: &str = "claude-code";
/// `routed_by` recorded when no routing decision was made.
pub const DEFAULT_ROUTED_BY: &str = "default";

fn default_provider_id() -> String {
    DEFAULT_PROVIDER_ID.to_string()
}

fn default_routed_by() -> String {
    DEFAULT_ROUTED_BY.to_string()
}

fn default_attempt() -> u32 {
    1
}

impl Default for AgentExecutionNode {
    /// A `running` first attempt on the default provider, with nil ids.
    /// Meant for struct-update syntax; use [`AgentExecutionNode::new`] to get
    /// a fresh id.
    fn default() -> Self {
        Self {
            id: Uuid::nil(),
            run_id: Uuid::nil(),
            task_id: Uuid::nil(),
            session_id: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
            cost_usd: 0.0,
            duration_secs: 0.0,
            status: AgentExecutionStatus::Running,
            tools_used: "{}".to_string(),
            files_modified: Vec::new(),
            commits: Vec::new(),
            persona_profile: String::new(),
            vector_json: None,
            report_json: None,
            execution_type: ExecutionType::default(),
            provider_id: default_provider_id(),
            model_requested: None,
            model: None,
            model_alias: None,
            routed_by: default_routed_by(),
            route_rule: None,
            fallback_reason: None,
            shadow_model: None,
            shadow_provider: None,
            task_class: None,
            attempt: default_attempt(),
            base_sha: None,
            tokens_in: None,
            tokens_out: None,
            tokens_cache_read: None,
            tokens_cache_write: None,
            cost_basis: None,
            num_turns: None,
            verification_json: None,
            verification_passed: None,
            routing_decision_id: None,
        }
    }
}

impl AgentExecutionNode {
    /// A fresh `running` first attempt for `task_id` within `run_id`.
    pub fn new(run_id: Uuid, task_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            run_id,
            task_id,
            ..Default::default()
        }
    }

    /// Optional string properties decided when the attempt is launched.
    fn launch_string_props(&self) -> Vec<(&'static str, String)> {
        let decision = self.routing_decision_id.map(|id| id.to_string());
        [
            ("routing_decision_id", &decision),
            ("model_requested", &self.model_requested),
            ("model_alias", &self.model_alias),
            ("route_rule", &self.route_rule),
            ("shadow_model", &self.shadow_model),
            ("shadow_provider", &self.shadow_provider),
            ("task_class", &self.task_class),
            ("base_sha", &self.base_sha),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.clone().map(|v| (name, v)))
        .collect()
    }

    /// Optional string properties known once the attempt has run.
    fn outcome_string_props(&self) -> Vec<(&'static str, String)> {
        [
            ("model", &self.model),
            ("fallback_reason", &self.fallback_reason),
            ("cost_basis", &self.cost_basis),
            ("verification_json", &self.verification_json),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.clone().map(|v| (name, v)))
        .collect()
    }

    /// Optional integer properties known once the attempt has run.
    fn outcome_int_props(&self) -> Vec<(&'static str, i64)> {
        [
            ("tokens_in", self.tokens_in),
            ("tokens_out", self.tokens_out),
            ("tokens_cache_read", self.tokens_cache_read),
            ("tokens_cache_write", self.tokens_cache_write),
            ("num_turns", self.num_turns.map(u64::from)),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|v| (name, v as i64)))
        .collect()
    }
}

/// Type of agent execution — distinguishes regular task runs from gate retries
/// and verification passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionType {
    /// Normal task execution by an agent.
    #[default]
    TaskAgent,
    /// Re-execution triggered by a quality gate failure.
    GateRetry,
    /// Verification pass (e.g., running tests after a fix).
    Verification,
}

impl std::fmt::Display for ExecutionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TaskAgent => write!(f, "task_agent"),
            Self::GateRetry => write!(f, "gate_retry"),
            Self::Verification => write!(f, "verification"),
        }
    }
}

impl ExecutionType {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "gate_retry" => Self::GateRetry,
            "verification" => Self::Verification,
            _ => Self::TaskAgent,
        }
    }
}

/// Status of an agent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentExecutionStatus {
    Running,
    Completed,
    Failed,
    Timeout,
    /// The agent was left `running` by a process that no longer exists (server
    /// restart, crash) or by a run that ended without closing it. Whether it
    /// finished is unknown: this is neither `completed` nor `failed`.
    Interrupted,
}

impl std::fmt::Display for AgentExecutionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Timeout => write!(f, "timeout"),
            Self::Interrupted => write!(f, "interrupted"),
        }
    }
}

impl AgentExecutionStatus {
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "timeout" => Self::Timeout,
            "interrupted" => Self::Interrupted,
            _ => Self::Running,
        }
    }
}

/// Groups of the cost breakdown (B35).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostGroupBy {
    /// By effective model (`unknown` when the provider did not say).
    Model,
    /// By provider instance.
    Provider,
    /// By task class at launch (`unclassified` when none).
    TaskClass,
    /// By the resolution level that picked the provider/model.
    RoutedBy,
}

impl CostGroupBy {
    /// Every group the breakdown accepts, as the query parameter spells it.
    pub const ALL: [&'static str; 4] = ["model", "provider", "task_class", "routed_by"];

    /// The group a query parameter names, `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "model" => Some(Self::Model),
            "provider" => Some(Self::Provider),
            "task_class" => Some(Self::TaskClass),
            "routed_by" => Some(Self::RoutedBy),
            _ => None,
        }
    }

    /// The key an execution falls under: the same defaults the Cypher applies.
    pub fn key_of(self, ae: &AgentExecutionNode) -> String {
        fn or_default(value: Option<&str>, default: &str) -> String {
            value
                .filter(|s| !s.is_empty())
                .unwrap_or(default)
                .to_string()
        }
        match self {
            Self::Model => or_default(ae.model.as_deref(), "unknown"),
            Self::Provider => or_default(Some(&ae.provider_id), DEFAULT_PROVIDER_ID),
            Self::TaskClass => or_default(ae.task_class.as_deref(), "unclassified"),
            Self::RoutedBy => or_default(Some(&ae.routed_by), DEFAULT_ROUTED_BY),
        }
    }
}

/// Most groups one breakdown returns: the key spaces are small (models,
/// instances, classes, levels), the cap only keeps a runaway query bounded.
pub const MAX_BREAKDOWN_GROUPS: usize = 1_000;

/// The two counters and the unknowns of a group: never summed together (A21).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CostCell {
    /// Real spend: basis `reported` or `priced` (or none, the pre-A21 shape).
    pub marginal_usd: f64,
    /// Subscription or free usage at its stated amount.
    pub notional_usd: f64,
    /// Executions whose cost has no amount (`unknown` basis): counted, not summed.
    pub unknown_count: u64,
}

/// Tokens of a group, `None` when no execution reported the figure.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TokenCell {
    pub input: Option<u64>,
    pub output: Option<u64>,
}

/// One group of the cost breakdown (B35).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CostBreakdownRow {
    /// The group's key (model name, provider id, task class or routing level).
    pub key: String,
    /// Executions in the group (still-`running` attempts left out).
    pub executions: u64,
    pub cost: CostCell,
    pub tokens: TokenCell,
    /// Share of the first attempts (`attempt = 1`) whose verification passed;
    /// `None` when the group has no first attempt.
    pub verified_first_attempt_rate: Option<f64>,
    /// Median of `duration_secs` (lower middle, as Cypher's `percentileDisc(0.5)`);
    /// `None` for an empty group.
    pub median_duration_secs: Option<f64>,
}

/// Whether an execution belongs to the breakdown of `[from, to)` (the mock's
/// side of the Cypher `WHERE`).
#[cfg(test)]
pub fn in_breakdown_window(
    ae: &AgentExecutionNode,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> bool {
    ae.status != AgentExecutionStatus::Running && ae.started_at >= from && ae.started_at < to
}

/// The breakdown of `executions` by `group_by`, in memory: what the Cypher of
/// [`Neo4jClient::agent_execution_cost_breakdown_impl`] computes, for the mock
/// store and for the tests that pin the semantics. Sorted by key.
#[cfg(test)]
pub fn breakdown<'a>(
    executions: impl IntoIterator<Item = &'a AgentExecutionNode>,
    group_by: CostGroupBy,
) -> Vec<CostBreakdownRow> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Acc {
        row: CostBreakdownRow,
        first_attempts: u64,
        verified_first: u64,
        durations: Vec<f64>,
    }

    let mut groups: BTreeMap<String, Acc> = BTreeMap::new();
    for ae in executions {
        let acc = groups.entry(group_by.key_of(ae)).or_default();
        acc.row.executions += 1;
        match ae.cost_basis.as_deref() {
            None | Some("reported") | Some("priced") => acc.row.cost.marginal_usd += ae.cost_usd,
            Some("free") | Some("subscription") => acc.row.cost.notional_usd += ae.cost_usd,
            Some(_) => acc.row.cost.unknown_count += 1,
        }
        if let Some(t) = ae.tokens_in {
            *acc.row.tokens.input.get_or_insert(0) += t;
        }
        if let Some(t) = ae.tokens_out {
            *acc.row.tokens.output.get_or_insert(0) += t;
        }
        if ae.attempt <= 1 {
            acc.first_attempts += 1;
            if ae.verification_passed == Some(true) {
                acc.verified_first += 1;
            }
        }
        acc.durations.push(ae.duration_secs);
    }
    groups
        .into_iter()
        .take(MAX_BREAKDOWN_GROUPS)
        .map(|(key, mut acc)| {
            acc.row.key = key;
            acc.row.verified_first_attempt_rate = (acc.first_attempts > 0)
                .then(|| acc.verified_first as f64 / acc.first_attempts as f64);
            acc.durations
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            acc.row.median_duration_secs = (!acc.durations.is_empty()).then(|| {
                // percentileDisc(0.5): the value at the ceil(n / 2)-th position.
                let n = acc.durations.len();
                acc.durations[n.div_ceil(2) - 1]
            });
            acc.row
        })
        .collect()
}

impl Neo4jClient {
    // ========================================================================
    // AgentExecution operations
    // ========================================================================

    /// Create an AgentExecution node and link it to the PlanRun and Task.
    ///
    /// Creates:
    /// - `(:AgentExecution)-[:PART_OF]->(:PlanRun)`
    /// - `(:AgentExecution)-[:EXECUTES]->(:Task)`
    pub async fn create_agent_execution_impl(&self, ae: &AgentExecutionNode) -> Result<()> {
        let mut cypher = String::from(
            r#"
            MATCH (r:PlanRun {run_id: $run_id})
            MATCH (t:Task {id: $task_id})
            CREATE (ae:AgentExecution {
                id: $id,
                run_id: $run_id,
                task_id: $task_id,
                session_id: $session_id,
                started_at: datetime($started_at),
                cost_usd: $cost_usd,
                duration_secs: $duration_secs,
                status: $status,
                tools_used: $tools_used,
                files_modified: $files_modified,
                commits: $commits,
                persona_profile: $persona,
                execution_type: $execution_type,
                provider_id: $provider_id,
                routed_by: $routed_by,
                attempt: $attempt
            })
            CREATE (ae)-[:PART_OF]->(r)
            CREATE (ae)-[:EXECUTES]->(t)
            "#,
        );

        // Optional properties are only written when present (absent = None on read).
        let mut string_props = ae.launch_string_props();
        string_props.extend(ae.outcome_string_props());
        let int_props = ae.outcome_int_props();
        let assignments: Vec<String> = string_props
            .iter()
            .map(|(name, _)| *name)
            .chain(int_props.iter().map(|(name, _)| *name))
            .map(|name| format!("ae.{name} = ${name}"))
            .collect();
        if !assignments.is_empty() {
            cypher.push_str("SET ");
            cypher.push_str(&assignments.join(", "));
        }

        let mut q = query(&cypher)
            .param("id", ae.id.to_string())
            .param("run_id", ae.run_id.to_string())
            .param("task_id", ae.task_id.to_string())
            .param(
                "session_id",
                ae.session_id.map(|s| s.to_string()).unwrap_or_default(),
            )
            .param("started_at", ae.started_at.to_rfc3339())
            .param("cost_usd", ae.cost_usd)
            .param("duration_secs", ae.duration_secs)
            .param("status", ae.status.to_string())
            .param("tools_used", ae.tools_used.clone())
            .param("files_modified", ae.files_modified.clone())
            .param("commits", ae.commits.clone())
            .param("persona", ae.persona_profile.clone())
            .param("execution_type", ae.execution_type.to_string())
            .param("provider_id", ae.provider_id.clone())
            .param("routed_by", ae.routed_by.clone())
            .param("attempt", ae.attempt as i64);

        for (name, value) in string_props {
            q = q.param(name, value);
        }
        for (name, value) in int_props {
            q = q.param(name, value);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Update an existing AgentExecution node with final results.
    pub async fn update_agent_execution_impl(&self, ae: &AgentExecutionNode) -> Result<()> {
        let mut cypher = String::from(
            r#"
            MATCH (ae:AgentExecution {id: $id})
            SET ae.cost_usd = $cost_usd,
                ae.duration_secs = $duration_secs,
                ae.status = $status,
                ae.tools_used = $tools_used,
                ae.files_modified = $files_modified,
                ae.commits = $commits
            "#,
        );

        if ae.completed_at.is_some() {
            cypher.push_str(", ae.completed_at = datetime($completed_at)");
        }

        if ae.vector_json.is_some() {
            // Use parameter for vector_json to avoid injection
            cypher.push_str(", ae.vector_json = $vector_json");
        }

        if ae.report_json.is_some() {
            cypher.push_str(", ae.report_json = $report_json");
        }

        if ae.verification_passed.is_some() {
            cypher.push_str(", ae.verification_passed = $verification_passed");
        }

        // Outcome of the attempt (effective model, usage, cost basis, checks):
        // written only when known, so closing a node never erases a value.
        // Launch-time properties (provider, routing, attempt, base_sha) are set
        // at creation and left untouched here.
        let outcome_strings = ae.outcome_string_props();
        let outcome_ints = ae.outcome_int_props();
        for name in outcome_strings
            .iter()
            .map(|(name, _)| *name)
            .chain(outcome_ints.iter().map(|(name, _)| *name))
        {
            cypher.push_str(&format!(", ae.{name} = ${name}"));
        }

        let mut q = query(&cypher)
            .param("id", ae.id.to_string())
            .param("cost_usd", ae.cost_usd)
            .param("duration_secs", ae.duration_secs)
            .param("status", ae.status.to_string())
            .param("tools_used", ae.tools_used.clone())
            .param("files_modified", ae.files_modified.clone())
            .param("commits", ae.commits.clone());

        if let Some(completed_at) = ae.completed_at {
            q = q.param("completed_at", completed_at.to_rfc3339());
        }

        if let Some(ref vector_json) = ae.vector_json {
            q = q.param("vector_json", vector_json.clone());
        }

        if let Some(ref report_json) = ae.report_json {
            q = q.param("report_json", report_json.clone());
        }

        if let Some(passed) = ae.verification_passed {
            q = q.param("verification_passed", passed);
        }

        for (name, value) in outcome_strings {
            q = q.param(name, value);
        }
        for (name, value) in outcome_ints {
            q = q.param(name, value);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Record the per-check verification result on one AgentExecution.
    ///
    /// Single property, no read-modify-write: the attempt may be closed
    /// concurrently, and a full-node update from a stale copy would restore
    /// `status = running`.
    pub async fn record_agent_execution_verification_impl(
        &self,
        agent_execution_id: Uuid,
        verification_json: &str,
    ) -> Result<()> {
        let mut cypher = String::from(
            r#"
            MATCH (ae:AgentExecution {id: $id})
            SET ae.verification_json = $verification_json
            "#,
        );
        let passed = crate::runner::routing::verification_passed(Some(verification_json));
        if passed.is_some() {
            cypher.push_str(", ae.verification_passed = $verification_passed");
        }
        let mut q = query(&cypher)
            .param("id", agent_execution_id.to_string())
            .param("verification_json", verification_json);
        if let Some(passed) = passed {
            q = q.param("verification_passed", passed);
        }

        self.graph.run(q).await?;
        Ok(())
    }

    /// Cost, tokens, first-attempt verification and median duration of the
    /// executions started in `[from, to)`, grouped by `group_by` (B35).
    ///
    /// One aggregate per group, computed in the graph: the marginal and
    /// notional amounts are summed in separate cells, an unknown cost is
    /// counted and never summed, so no two bases ever meet in one figure.
    /// Still-`running` attempts have no cost yet and are left out. Bounded by
    /// the time window (`started_at` is indexed) and a cap on the groups.
    pub async fn agent_execution_cost_breakdown_impl(
        &self,
        group_by: CostGroupBy,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<CostBreakdownRow>> {
        let key = match group_by {
            CostGroupBy::Model => "CASE WHEN ae.model IS NULL OR ae.model = '' THEN 'unknown' ELSE ae.model END",
            CostGroupBy::Provider => "CASE WHEN ae.provider_id IS NULL OR ae.provider_id = '' THEN 'claude-code' ELSE ae.provider_id END",
            CostGroupBy::TaskClass => "CASE WHEN ae.task_class IS NULL OR ae.task_class = '' THEN 'unclassified' ELSE ae.task_class END",
            CostGroupBy::RoutedBy => "CASE WHEN ae.routed_by IS NULL OR ae.routed_by = '' THEN 'default' ELSE ae.routed_by END",
        };
        let cypher = format!(
            r#"
            MATCH (ae:AgentExecution)
            WHERE ae.started_at >= datetime($from) AND ae.started_at < datetime($to)
              AND ae.status <> 'running'
            WITH ae, {key} AS key,
                 CASE WHEN ae.cost_basis IS NULL OR ae.cost_basis IN ['reported', 'priced']
                      THEN coalesce(ae.cost_usd, 0.0) ELSE 0.0 END AS marginal,
                 CASE WHEN ae.cost_basis IN ['free', 'subscription']
                      THEN coalesce(ae.cost_usd, 0.0) ELSE 0.0 END AS notional,
                 CASE WHEN ae.cost_basis IS NULL OR ae.cost_basis IN ['reported', 'priced', 'free', 'subscription']
                      THEN 0 ELSE 1 END AS unknown
            RETURN key,
                   count(ae) AS executions,
                   sum(marginal) AS marginal_usd,
                   sum(notional) AS notional_usd,
                   sum(unknown) AS unknown_count,
                   count(ae.tokens_in) AS with_input,
                   sum(coalesce(ae.tokens_in, 0)) AS input_tokens,
                   count(ae.tokens_out) AS with_output,
                   sum(coalesce(ae.tokens_out, 0)) AS output_tokens,
                   sum(CASE WHEN coalesce(ae.attempt, 1) = 1 THEN 1 ELSE 0 END) AS first_attempts,
                   sum(CASE WHEN coalesce(ae.attempt, 1) = 1 AND ae.verification_passed = true THEN 1 ELSE 0 END) AS verified_first,
                   percentileDisc(ae.duration_secs, 0.5) AS median_duration_secs
            ORDER BY key ASC
            LIMIT {MAX_BREAKDOWN_GROUPS}
            "#
        );
        let q = query(&cypher)
            .param("from", from.to_rfc3339())
            .param("to", to.to_rfc3339());

        let mut result = self.graph.execute(q).await?;
        let mut rows = Vec::new();
        while let Some(row) = result.next().await? {
            let count = |name: &str| row.get::<i64>(name).unwrap_or(0).max(0) as u64;
            let with_input = count("with_input");
            let with_output = count("with_output");
            let first_attempts = count("first_attempts");
            rows.push(CostBreakdownRow {
                key: row.get("key")?,
                executions: count("executions"),
                cost: CostCell {
                    marginal_usd: row.get("marginal_usd").unwrap_or(0.0),
                    notional_usd: row.get("notional_usd").unwrap_or(0.0),
                    unknown_count: count("unknown_count"),
                },
                tokens: TokenCell {
                    input: (with_input > 0).then(|| count("input_tokens")),
                    output: (with_output > 0).then(|| count("output_tokens")),
                },
                verified_first_attempt_rate: (first_attempts > 0)
                    .then(|| count("verified_first") as f64 / first_attempts as f64),
                median_duration_secs: row.get::<f64>("median_duration_secs").ok(),
            });
        }
        Ok(rows)
    }

    /// Get all AgentExecution nodes for a given PlanRun.
    pub async fn get_agent_executions_for_run_impl(
        &self,
        run_id: Uuid,
    ) -> Result<Vec<AgentExecutionNode>> {
        let q = query(
            r#"
            MATCH (ae:AgentExecution {run_id: $run_id})
            RETURN ae
            ORDER BY ae.started_at ASC
            "#,
        )
        .param("run_id", run_id.to_string());

        let mut result = self.graph.execute(q).await?;
        let mut executions = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("ae")?;
            executions.push(self.node_to_agent_execution(&node)?);
        }
        Ok(executions)
    }

    /// Every AgentExecution whose status is still `running`, oldest first.
    pub async fn list_running_agent_executions_impl(&self) -> Result<Vec<AgentExecutionNode>> {
        let q = query(
            r#"
            MATCH (ae:AgentExecution {status: 'running'})
            RETURN ae
            ORDER BY ae.started_at ASC
            "#,
        );

        let mut result = self.graph.execute(q).await?;
        let mut executions = Vec::new();
        while let Some(row) = result.next().await? {
            let node: neo4rs::Node = row.get("ae")?;
            executions.push(self.node_to_agent_execution(&node)?);
        }
        Ok(executions)
    }

    /// Create a USED_SKILL relationship from an AgentExecution to a Skill.
    pub async fn create_used_skill_relation_impl(
        &self,
        agent_execution_id: Uuid,
        skill_id: Uuid,
        result: &str,
    ) -> Result<()> {
        let q = query(
            r#"
            MATCH (ae:AgentExecution {id: $ae_id})
            MATCH (s:Skill {id: $skill_id})
            MERGE (ae)-[r:USED_SKILL]->(s)
            SET r.result = $result,
                r.timestamp = datetime()
            "#,
        )
        .param("ae_id", agent_execution_id.to_string())
        .param("skill_id", skill_id.to_string())
        .param("result", result);

        self.graph.run(q).await?;
        Ok(())
    }

    /// Convert a Neo4j node to AgentExecutionNode.
    fn node_to_agent_execution(&self, node: &neo4rs::Node) -> Result<AgentExecutionNode> {
        let id: String = node.get("id")?;
        let run_id: String = node.get("run_id")?;
        let task_id: String = node.get("task_id")?;
        let session_id: Option<String> = node.get("session_id").ok();
        let started_at: String = node.get("started_at")?;
        let completed_at: Option<String> = node.get("completed_at").ok();
        let status: String = node.get("status")?;

        let files_modified: Vec<String> = node.get("files_modified").unwrap_or_default();
        let commits: Vec<String> = node.get("commits").unwrap_or_default();

        // A22 properties are additive: a missing (or empty) one reads as its default.
        let opt_string = |name: &str| -> Option<String> {
            node.get::<String>(name).ok().filter(|s| !s.is_empty())
        };
        let opt_u64 = |name: &str| -> Option<u64> {
            node.get::<i64>(name)
                .ok()
                .and_then(|v| u64::try_from(v).ok())
        };

        Ok(AgentExecutionNode {
            id: id.parse()?,
            run_id: run_id.parse()?,
            task_id: task_id.parse()?,
            session_id: session_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok()),
            started_at: started_at.parse()?,
            completed_at: completed_at.and_then(|s| s.parse().ok()),
            cost_usd: node.get("cost_usd").unwrap_or(0.0),
            duration_secs: node.get("duration_secs").unwrap_or(0.0),
            status: AgentExecutionStatus::from_str_lossy(&status),
            tools_used: node.get("tools_used").unwrap_or_default(),
            files_modified,
            commits,
            persona_profile: node.get("persona_profile").unwrap_or_default(),
            vector_json: node.get("vector_json").ok(),
            report_json: node.get("report_json").ok(),
            execution_type: node
                .get::<String>("execution_type")
                .map(|s| ExecutionType::from_str_lossy(&s))
                .unwrap_or_default(),
            provider_id: opt_string("provider_id").unwrap_or_else(default_provider_id),
            model_requested: opt_string("model_requested"),
            model: opt_string("model"),
            model_alias: opt_string("model_alias"),
            routed_by: opt_string("routed_by").unwrap_or_else(default_routed_by),
            route_rule: opt_string("route_rule"),
            fallback_reason: opt_string("fallback_reason"),
            shadow_model: opt_string("shadow_model"),
            shadow_provider: opt_string("shadow_provider"),
            task_class: opt_string("task_class"),
            attempt: opt_u64("attempt")
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v > 0)
                .unwrap_or_else(default_attempt),
            base_sha: opt_string("base_sha"),
            tokens_in: opt_u64("tokens_in"),
            tokens_out: opt_u64("tokens_out"),
            tokens_cache_read: opt_u64("tokens_cache_read"),
            tokens_cache_write: opt_u64("tokens_cache_write"),
            cost_basis: opt_string("cost_basis"),
            num_turns: opt_u64("num_turns").and_then(|v| u32::try_from(v).ok()),
            verification_json: opt_string("verification_json"),
            verification_passed: node.get::<bool>("verification_passed").ok(),
            routing_decision_id: opt_string("routing_decision_id")
                .and_then(|s| Uuid::parse_str(&s).ok()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_agent_execution(
        report_json: Option<String>,
        vector_json: Option<String>,
    ) -> AgentExecutionNode {
        AgentExecutionNode {
            id: Uuid::new_v4(),
            run_id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            session_id: Some(Uuid::new_v4()),
            started_at: Utc::now(),
            completed_at: Some(Utc::now()),
            cost_usd: 0.05,
            duration_secs: 12.5,
            status: AgentExecutionStatus::Completed,
            tools_used: "note,code".to_string(),
            files_modified: vec!["src/main.rs".to_string()],
            commits: vec!["abc123".to_string()],
            persona_profile: "test-profile".to_string(),
            vector_json,
            report_json,
            execution_type: ExecutionType::TaskAgent,
            ..Default::default()
        }
    }

    /// A node with every A22 field set to a non-default value.
    fn make_fully_recorded_execution() -> AgentExecutionNode {
        AgentExecutionNode {
            provider_id: "native".to_string(),
            model_requested: Some("cheap".to_string()),
            model: Some("some-model-v2".to_string()),
            model_alias: Some("cheap".to_string()),
            routed_by: "project_rule".to_string(),
            route_rule: Some("runner.simple".to_string()),
            fallback_reason: Some("endpoint_unhealthy".to_string()),
            shadow_model: Some("other-model".to_string()),
            shadow_provider: Some("other-provider".to_string()),
            task_class: Some("simple".to_string()),
            attempt: 2,
            base_sha: Some("0123abcd".to_string()),
            tokens_in: Some(1200),
            tokens_out: Some(340),
            tokens_cache_read: Some(5000),
            tokens_cache_write: Some(64),
            cost_basis: Some("reported".to_string()),
            num_turns: Some(7),
            verification_json: Some(r#"{"build":"pass"}"#.to_string()),
            verification_passed: Some(true),
            routing_decision_id: Some(Uuid::new_v4()),
            ..make_agent_execution(None, None)
        }
    }

    fn assert_record_fields_eq(a: &AgentExecutionNode, b: &AgentExecutionNode) {
        assert_eq!(a.provider_id, b.provider_id);
        assert_eq!(a.model_requested, b.model_requested);
        assert_eq!(a.model, b.model);
        assert_eq!(a.model_alias, b.model_alias);
        assert_eq!(a.routed_by, b.routed_by);
        assert_eq!(a.route_rule, b.route_rule);
        assert_eq!(a.fallback_reason, b.fallback_reason);
        assert_eq!(a.shadow_model, b.shadow_model);
        assert_eq!(a.shadow_provider, b.shadow_provider);
        assert_eq!(a.task_class, b.task_class);
        assert_eq!(a.attempt, b.attempt);
        assert_eq!(a.base_sha, b.base_sha);
        assert_eq!(a.tokens_in, b.tokens_in);
        assert_eq!(a.tokens_out, b.tokens_out);
        assert_eq!(a.tokens_cache_read, b.tokens_cache_read);
        assert_eq!(a.tokens_cache_write, b.tokens_cache_write);
        assert_eq!(a.cost_basis, b.cost_basis);
        assert_eq!(a.num_turns, b.num_turns);
        assert_eq!(a.verification_json, b.verification_json);
        assert_eq!(a.verification_passed, b.verification_passed);
        assert_eq!(a.routing_decision_id, b.routing_decision_id);
    }

    #[test]
    fn test_agent_execution_record_fields_serde_roundtrip() {
        let ae = make_fully_recorded_execution();
        let json = serde_json::to_string(&ae).unwrap();
        let back: AgentExecutionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, ae.id);
        assert_record_fields_eq(&back, &ae);
        assert_eq!(back.attempt, 2);
        assert_eq!(back.provider_id, "native");
        assert_eq!(back.tokens_cache_read, Some(5000));
    }

    #[test]
    fn test_agent_execution_legacy_json_reads_with_defaults() {
        // Shape written before the A22 fields existed.
        let legacy = serde_json::json!({
            "id": Uuid::new_v4(),
            "run_id": Uuid::new_v4(),
            "task_id": Uuid::new_v4(),
            "session_id": null,
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": null,
            "cost_usd": 0.25,
            "duration_secs": 3.0,
            "status": "completed",
            "tools_used": "{}",
            "files_modified": [],
            "commits": [],
            "persona_profile": "simple",
            "vector_json": null,
            "report_json": null
        })
        .to_string();
        let ae: AgentExecutionNode = serde_json::from_str(&legacy).unwrap();
        assert_eq!(ae.attempt, 1);
        assert_eq!(ae.provider_id, "claude-code");
        assert_eq!(ae.routed_by, "default");
        assert!(ae.model_requested.is_none());
        assert!(ae.model.is_none());
        assert!(ae.model_alias.is_none());
        assert!(ae.route_rule.is_none());
        assert!(ae.fallback_reason.is_none());
        assert!(ae.shadow_model.is_none());
        assert!(ae.shadow_provider.is_none());
        assert!(ae.task_class.is_none());
        assert!(ae.base_sha.is_none());
        assert!(ae.tokens_in.is_none());
        assert!(ae.tokens_out.is_none());
        assert!(ae.tokens_cache_read.is_none());
        assert!(ae.tokens_cache_write.is_none());
        assert!(ae.cost_basis.is_none());
        assert!(ae.num_turns.is_none());
        assert!(ae.verification_json.is_none());
        assert!(ae.verification_passed.is_none());
        assert!(
            ae.routing_decision_id.is_none(),
            "a node written before routing decisions reads as None"
        );
    }

    #[test]
    fn test_routing_decision_id_is_written_at_launch_and_only_when_set() {
        let mut ae = AgentExecutionNode::default();
        assert!(!ae
            .launch_string_props()
            .iter()
            .any(|(name, _)| *name == "routing_decision_id"));
        let id = Uuid::new_v4();
        ae.routing_decision_id = Some(id);
        let props = ae.launch_string_props();
        let written = props
            .iter()
            .find(|(name, _)| *name == "routing_decision_id");
        assert_eq!(
            written.map(|(_, v)| v.as_str()),
            Some(id.to_string().as_str())
        );
    }

    #[test]
    fn test_agent_execution_default_and_new() {
        let d = AgentExecutionNode::default();
        assert_eq!(d.attempt, 1);
        assert_eq!(d.provider_id, DEFAULT_PROVIDER_ID);
        assert_eq!(d.routed_by, DEFAULT_ROUTED_BY);
        assert_eq!(d.status, AgentExecutionStatus::Running);
        assert!(d.completed_at.is_none());

        let (run_id, task_id) = (Uuid::new_v4(), Uuid::new_v4());
        let n = AgentExecutionNode::new(run_id, task_id);
        assert_eq!(n.run_id, run_id);
        assert_eq!(n.task_id, task_id);
        assert!(!n.id.is_nil());
    }

    #[test]
    fn test_agent_execution_optional_props_skip_none() {
        let bare = make_agent_execution(None, None);
        assert!(bare.launch_string_props().is_empty());
        assert!(bare.outcome_string_props().is_empty());
        assert!(bare.outcome_int_props().is_empty());

        let full = make_fully_recorded_execution();
        assert_eq!(full.launch_string_props().len(), 8);
        assert_eq!(full.outcome_string_props().len(), 4);
        let ints = full.outcome_int_props();
        assert_eq!(ints.len(), 5);
        assert!(ints.contains(&("num_turns", 7)));
        assert!(ints.contains(&("tokens_in", 1200)));
    }

    #[tokio::test]
    async fn test_agent_execution_record_fields_mock_roundtrip() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::traits::GraphStore;

        let g = MockGraphStore::new();
        let ae = make_fully_recorded_execution();
        g.create_agent_execution(&ae).await.unwrap();

        let read = g.get_agent_executions_for_run(ae.run_id).await.unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].id, ae.id);
        assert_record_fields_eq(&read[0], &ae);

        // An update keeps the record fields it is given.
        let mut closed = ae.clone();
        closed.status = AgentExecutionStatus::Failed;
        closed.tokens_out = Some(999);
        g.update_agent_execution(&closed).await.unwrap();
        let read = g.get_agent_executions_for_run(ae.run_id).await.unwrap();
        assert_eq!(read[0].status, AgentExecutionStatus::Failed);
        assert_record_fields_eq(&read[0], &closed);
    }

    #[test]
    fn test_agent_execution_node_serialize_with_report_json() {
        let ae = make_agent_execution(
            Some(r#"{"summary":"ok"}"#.to_string()),
            Some(r#"{"vec":[1,2]}"#.to_string()),
        );
        let json = serde_json::to_string(&ae).unwrap();
        assert!(json.contains("report_json"));
        // report_json is a String field, so the inner JSON is escaped in the outer JSON
        assert!(json.contains("summary"));
        assert!(json.contains("vector_json"));
        assert!(json.contains("vec"));
    }

    #[test]
    fn test_agent_execution_node_serialize_without_report_json() {
        let ae = make_agent_execution(None, None);
        let json = serde_json::to_string(&ae).unwrap();
        assert!(json.contains("\"report_json\":null"));
        assert!(json.contains("\"vector_json\":null"));
    }

    #[test]
    fn test_agent_execution_node_roundtrip() {
        let ae = make_agent_execution(
            Some(r#"{"tasks_completed":3}"#.to_string()),
            Some(r#"{"energy":0.9}"#.to_string()),
        );
        let json = serde_json::to_string(&ae).unwrap();
        let deserialized: AgentExecutionNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.report_json, ae.report_json);
        assert_eq!(deserialized.vector_json, ae.vector_json);
        assert_eq!(deserialized.id, ae.id);
        assert_eq!(deserialized.cost_usd, ae.cost_usd);
        assert_eq!(deserialized.status, ae.status);
    }

    #[test]
    fn test_agent_execution_status_display() {
        assert_eq!(AgentExecutionStatus::Running.to_string(), "running");
        assert_eq!(AgentExecutionStatus::Completed.to_string(), "completed");
        assert_eq!(AgentExecutionStatus::Failed.to_string(), "failed");
        assert_eq!(AgentExecutionStatus::Timeout.to_string(), "timeout");
    }

    #[test]
    fn test_agent_execution_status_from_str_lossy() {
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("completed"),
            AgentExecutionStatus::Completed
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("failed"),
            AgentExecutionStatus::Failed
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("timeout"),
            AgentExecutionStatus::Timeout
        );
        // Unknown strings default to Running
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("running"),
            AgentExecutionStatus::Running
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy("unknown"),
            AgentExecutionStatus::Running
        );
        assert_eq!(
            AgentExecutionStatus::from_str_lossy(""),
            AgentExecutionStatus::Running
        );
    }

    #[test]
    fn test_agent_execution_node_without_session_id() {
        let mut ae = make_agent_execution(None, None);
        ae.session_id = None;
        ae.completed_at = None;
        let json = serde_json::to_string(&ae).unwrap();
        let deserialized: AgentExecutionNode = serde_json::from_str(&json).unwrap();
        assert!(deserialized.session_id.is_none());
        assert!(deserialized.completed_at.is_none());
    }

    #[test]
    fn test_agent_execution_status_serialize_roundtrip() {
        for status in &[
            AgentExecutionStatus::Running,
            AgentExecutionStatus::Completed,
            AgentExecutionStatus::Failed,
            AgentExecutionStatus::Timeout,
        ] {
            let json = serde_json::to_string(status).unwrap();
            let deserialized: AgentExecutionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, deserialized);
        }
    }

    #[test]
    fn test_agent_execution_node_debug() {
        let ae = make_agent_execution(Some("report".to_string()), None);
        let debug = format!("{:?}", ae);
        assert!(debug.contains("AgentExecutionNode"));
        assert!(debug.contains("report_json"));
    }

    #[test]
    fn test_agent_execution_node_clone() {
        let ae = make_agent_execution(Some("report".to_string()), Some("vector".to_string()));
        let cloned = ae.clone();
        assert_eq!(cloned.id, ae.id);
        assert_eq!(cloned.report_json, ae.report_json);
        assert_eq!(cloned.vector_json, ae.vector_json);
    }

    #[test]
    fn test_execution_type_display() {
        assert_eq!(ExecutionType::TaskAgent.to_string(), "task_agent");
        assert_eq!(ExecutionType::GateRetry.to_string(), "gate_retry");
        assert_eq!(ExecutionType::Verification.to_string(), "verification");
    }

    #[test]
    fn test_execution_type_from_str_lossy() {
        assert_eq!(
            ExecutionType::from_str_lossy("gate_retry"),
            ExecutionType::GateRetry
        );
        assert_eq!(
            ExecutionType::from_str_lossy("verification"),
            ExecutionType::Verification
        );
        assert_eq!(
            ExecutionType::from_str_lossy("task_agent"),
            ExecutionType::TaskAgent
        );
        // Unknown defaults to TaskAgent
        assert_eq!(
            ExecutionType::from_str_lossy("unknown"),
            ExecutionType::TaskAgent
        );
        assert_eq!(ExecutionType::from_str_lossy(""), ExecutionType::TaskAgent);
    }

    #[test]
    fn test_execution_type_default() {
        let default: ExecutionType = Default::default();
        assert_eq!(default, ExecutionType::TaskAgent);
    }

    #[test]
    fn test_execution_type_serde_roundtrip() {
        for variant in &[
            ExecutionType::TaskAgent,
            ExecutionType::GateRetry,
            ExecutionType::Verification,
        ] {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: ExecutionType = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, deserialized);
        }
    }

    // ── Cost breakdown (B35) ───────────────────────────────────────────────

    fn execution(
        provider: &str,
        model: Option<&str>,
        class: Option<&str>,
        routed_by: &str,
        usd: f64,
        basis: Option<&str>,
    ) -> AgentExecutionNode {
        let mut ae = AgentExecutionNode::new(Uuid::nil(), Uuid::new_v4());
        ae.status = AgentExecutionStatus::Completed;
        ae.provider_id = provider.into();
        ae.model = model.map(str::to_string);
        ae.task_class = class.map(str::to_string);
        ae.routed_by = routed_by.into();
        ae.cost_usd = usd;
        ae.cost_basis = basis.map(str::to_string);
        ae.duration_secs = 10.0;
        ae
    }

    #[test]
    fn a_group_by_is_parsed_from_the_four_names_only() {
        for name in CostGroupBy::ALL {
            assert!(CostGroupBy::parse(name).is_some(), "{name}");
        }
        assert_eq!(CostGroupBy::parse("session"), None);
        assert_eq!(CostGroupBy::parse(""), None);
        assert_eq!(CostGroupBy::parse("Model"), None, "exact spelling");
    }

    /// (B35) Three bases, four cells: the marginal and the notional sums never
    /// meet, and an unknown cost is a count, not a zero in a sum. Old nodes
    /// (no model) fall under `unknown`; no basis at all is a reported cost.
    #[test]
    fn a_breakdown_never_adds_two_bases_in_one_cell_and_files_old_nodes_under_unknown() {
        let mut reported = execution(
            "claude-code",
            Some("opus"),
            Some("complex"),
            "default",
            2.0,
            Some("reported"),
        );
        reported.tokens_in = Some(100);
        reported.tokens_out = Some(10);
        let mut priced = execution(
            "deepseek",
            Some("ds-chat"),
            Some("simple"),
            "project_rule",
            0.5,
            Some("priced"),
        );
        priced.tokens_in = Some(40);
        let sub = execution(
            "claude-code",
            Some("opus"),
            Some("simple"),
            "default",
            3.0,
            Some("subscription"),
        );
        let free = execution(
            "local",
            Some("llama"),
            Some("simple"),
            "run",
            0.0,
            Some("free"),
        );
        // A fictitious amount on an unknown basis is never summed anywhere.
        let unknown = execution("local", Some("llama"), None, "run", 99.0, Some("unknown"));
        let mut old = execution("claude-code", None, None, "default", 0.25, None);
        old.attempt = 1;
        old.duration_secs = 30.0;

        let rows = breakdown(
            [&reported, &priced, &sub, &free, &unknown, &old],
            CostGroupBy::Model,
        );
        let by_key = |k: &str| rows.iter().find(|r| r.key == k).cloned().expect(k);
        assert_eq!(
            rows.iter().map(|r| r.key.as_str()).collect::<Vec<_>>(),
            ["ds-chat", "llama", "opus", "unknown"]
        );
        let opus = by_key("opus");
        assert_eq!(opus.executions, 2);
        assert_eq!(
            opus.cost,
            CostCell {
                marginal_usd: 2.0,
                notional_usd: 3.0,
                unknown_count: 0
            }
        );
        assert_eq!(
            opus.tokens,
            TokenCell {
                input: Some(100),
                output: Some(10)
            }
        );
        let llama = by_key("llama");
        assert_eq!(
            llama.cost,
            CostCell {
                marginal_usd: 0.0,
                notional_usd: 0.0,
                unknown_count: 1
            },
            "99 USD of unknown basis counted, never summed"
        );
        assert_eq!(
            llama.tokens,
            TokenCell {
                input: None,
                output: None
            },
            "nothing reported: unknown, not zero"
        );
        let old_nodes = by_key("unknown");
        assert_eq!(old_nodes.executions, 1);
        assert_eq!(
            old_nodes.cost.marginal_usd, 0.25,
            "no basis at all: a reported cost"
        );
        assert_eq!(old_nodes.median_duration_secs, Some(30.0));
        assert_eq!(
            by_key("ds-chat").tokens,
            TokenCell {
                input: Some(40),
                output: None
            }
        );

        let by_provider = breakdown(
            [&reported, &priced, &sub, &free, &unknown, &old],
            CostGroupBy::Provider,
        );
        assert_eq!(
            by_provider
                .iter()
                .map(|r| r.key.as_str())
                .collect::<Vec<_>>(),
            ["claude-code", "deepseek", "local"]
        );
        assert_eq!(
            by_provider[0].cost,
            CostCell {
                marginal_usd: 2.25,
                notional_usd: 3.0,
                unknown_count: 0
            }
        );
        let by_class = breakdown(
            [&reported, &priced, &sub, &free, &unknown, &old],
            CostGroupBy::TaskClass,
        );
        assert_eq!(
            by_class
                .iter()
                .map(|r| (r.key.as_str(), r.executions))
                .collect::<Vec<_>>(),
            [("complex", 1), ("simple", 3), ("unclassified", 2)]
        );
        let by_route = breakdown(
            [&reported, &priced, &sub, &free, &unknown, &old],
            CostGroupBy::RoutedBy,
        );
        assert_eq!(
            by_route.iter().map(|r| r.key.as_str()).collect::<Vec<_>>(),
            ["default", "project_rule", "run"]
        );
    }

    #[test]
    fn a_breakdown_rates_first_attempts_and_takes_the_lower_median() {
        let mut first_ok = execution("p", Some("m"), None, "default", 0.1, Some("reported"));
        first_ok.verification_passed = Some(true);
        first_ok.duration_secs = 5.0;
        let mut first_ko = execution("p", Some("m"), None, "default", 0.1, Some("reported"));
        first_ko.verification_passed = Some(false);
        first_ko.duration_secs = 50.0;
        let mut first_unverified =
            execution("p", Some("m"), None, "default", 0.1, Some("reported"));
        first_unverified.duration_secs = 20.0;
        let mut retry = execution("p", Some("m"), None, "default", 0.1, Some("reported"));
        retry.attempt = 2;
        retry.verification_passed = Some(true);
        retry.duration_secs = 40.0;

        let rows = breakdown(
            [&first_ok, &first_ko, &first_unverified, &retry],
            CostGroupBy::Provider,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].executions, 4);
        assert_eq!(
            rows[0].verified_first_attempt_rate,
            Some(1.0 / 3.0),
            "a retry's pass does not count as first-attempt"
        );
        assert_eq!(
            rows[0].median_duration_secs,
            Some(20.0),
            "lower middle of [5, 20, 40, 50]"
        );

        let only_retries = breakdown([&retry], CostGroupBy::Provider);
        assert_eq!(only_retries[0].verified_first_attempt_rate, None);
        assert!(breakdown(std::iter::empty(), CostGroupBy::Provider).is_empty());
    }

    #[test]
    fn the_breakdown_window_keeps_closed_attempts_started_in_it() {
        use chrono::TimeZone;
        let from = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap();
        let mut inside = execution("p", None, None, "default", 0.0, None);
        inside.started_at = from;
        assert!(in_breakdown_window(&inside, from, to), "from is inclusive");
        let mut at_end = inside.clone();
        at_end.started_at = to;
        assert!(!in_breakdown_window(&at_end, from, to), "to is exclusive");
        let mut running = inside.clone();
        running.status = AgentExecutionStatus::Running;
        assert!(!in_breakdown_window(&running, from, to), "no cost yet");
    }

    #[tokio::test]
    async fn recording_a_verification_sets_the_flat_verdict_on_the_mock() {
        use crate::neo4j::mock::MockGraphStore;
        use crate::neo4j::traits::GraphStore;
        let g = MockGraphStore::new();
        let ae = make_agent_execution(None, None);
        g.create_agent_execution(&ae).await.unwrap();
        g.record_agent_execution_verification(ae.id, r#"{"passed": false, "checks": {}}"#)
            .await
            .unwrap();
        let read = g.get_agent_executions_for_run(ae.run_id).await.unwrap();
        assert_eq!(read[0].verification_passed, Some(false));
    }
}
