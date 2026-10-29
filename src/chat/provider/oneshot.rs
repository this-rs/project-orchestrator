//! One-shot sessions outside the chat (B18): the feature-graph proposal and any other
//! single-turn utility call.
//!
//! A one-shot runs ONE turn on the instance the registry resolves for the project's
//! executor role, through the same resolver ([`resolver::resolve`], A16: explicit choice,
//! then project rule, global rule, configured default, built-in Claude Code) and the same
//! provider factory as every session — never a client built by hand on the Claude CLI.
//! The project's consent (A28) and the health of the instance are checked before anything
//! is sent; a refusal is typed, never worked around.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::sync::Arc;
use std::time::Duration;

use nexus_claude::agent::{
    AgentProvider, HealthStatus, PolicyMode, ProviderError, SessionSpec, SystemPromptMode,
    SystemPromptSpec, ToolPolicy, TurnInput, TurnSummary,
};

use super::resolver::{self, Candidate, ResolveError, ResolveInput, Role, RoutedBy};
use super::{catalog, native_factory, settings, store};
use crate::chat::agent_runtime::ProviderSource;
#[allow(unused_imports)]
use crate::neo4j::traits::*;
use crate::neo4j::GraphStore;

/// How long the health preflight of a one-shot may take.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);

/// What a one-shot runs on, and which level decided it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OneShotChoice {
    /// Instance identifier (registry key).
    pub provider_id: String,
    /// Model, when a level named one (`None` = the instance's default).
    pub model: Option<String>,
    /// The level that produced the choice.
    pub routed_by: RoutedBy,
    /// Why a higher level was skipped (`routed_by = fallback`).
    pub fallback_reason: Option<String>,
}

/// Resolves what a one-shot of `project_slug` runs on.
///
/// `requested` is an explicit choice that wins when given: a provider (and model), or a
/// model alone, which then rides on the instance the rules resolve. It is an executor
/// resolution: a rule naming an instance that is not allowed here is skipped with the
/// reason recorded, and nothing configured means the built-in Claude Code when its CLI is
/// on the server (`claude_cli_present`), else the only other usable instance, else a typed
/// [`ResolveError::NoProvider`].
pub async fn resolve(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
    requested: (Option<&str>, Option<&str>),
    claude_cli_present: bool,
) -> anyhow::Result<OneShotChoice> {
    let (requested_provider, requested_model) = requested;
    let instances = store::instances(graph).await?;
    let consents = store::consents_for(graph, project_slug).await?;
    let global = store::roles(graph, settings::GLOBAL).await?;
    let project = match project_slug {
        Some(slug) => store::roles(graph, &settings::project_scope(slug)).await?,
        None => Default::default(),
    };
    let aliases = store::aliases(graph).await?;
    let store_catalog =
        catalog::StoreCatalog::new(instances, &consents).with_claude_cli(claude_cli_present);
    let mut input: ResolveInput<'_> = catalog::resolve_input(
        Role::Executor,
        requested_provider,
        None,
        None,
        &global,
        &project,
        &aliases,
    );
    if let (Some(request), Some(model)) = (input.request.as_mut(), requested_model) {
        request.model = Some(model.to_string());
    }
    let choice = resolver::resolve(&input, &store_catalog)
        .map_err(|e: ResolveError| anyhow::Error::new(e.in_scope(project_slug)))?;
    Ok(OneShotChoice {
        provider_id: choice.provider_id,
        // A model named alone rides on whatever instance the rules resolved.
        model: requested_model.map(str::to_string).or(choice.model),
        routed_by: choice.routed_by,
        fallback_reason: choice.fallback_reason,
    })
}

/// Same, from the candidate a level hands over (`None` = nothing requested).
pub async fn resolve_candidate(
    graph: &dyn GraphStore,
    project_slug: Option<&str>,
    requested: Option<&Candidate>,
    claude_cli_present: bool,
) -> anyhow::Result<OneShotChoice> {
    resolve(
        graph,
        project_slug,
        (
            requested.map(|c| c.provider_id.as_str()),
            requested.and_then(|c| c.model.as_deref()),
        ),
        claude_cli_present,
    )
    .await
}

/// The instance `provider_id` as the registry builds it: the built-in one from
/// `builtin`, a stored one from its record through the provider factory (no vault: an
/// instance keyed in the vault answers `credentials_locked`, as it would with the vault
/// locked). An id nobody registered is a typed [`ResolveError::UnknownProvider`].
pub async fn provider_for(
    graph: &dyn GraphStore,
    provider_id: &str,
    builtin: &dyn ProviderSource,
) -> anyhow::Result<Arc<dyn AgentProvider>> {
    if let Some(provider) = builtin.get(provider_id) {
        return Ok(provider);
    }
    let record = store::instance(graph, provider_id).await?.ok_or_else(|| {
        anyhow::Error::new(ResolveError::UnknownProvider(provider_id.to_string()))
    })?;
    native_factory::build_native_provider(&record, None).map_err(anyhow::Error::new)
}

/// The one turn of a one-shot.
#[derive(Debug, Clone)]
pub struct OneShotTurn {
    /// Working directory of the session.
    pub cwd: std::path::PathBuf,
    /// The system prompt, replacing the provider's own.
    pub system_prompt: String,
    /// The user turn.
    pub prompt: String,
}

/// Opens ONE session on the resolved instance, runs one turn to its end and closes it.
///
/// The health of the instance is checked first (bounded), as the chat manager does at
/// every opening (A30): an unavailable instance is the typed error its health reports.
/// The session runs in trust with one turn, in the same clean environment as every agent
/// process this server starts (A33). A permission request met on the way is denied by
/// [`nexus_claude::agent::run_turn`] (nobody is there to answer).
pub async fn run(
    graph: &dyn GraphStore,
    choice: &OneShotChoice,
    turn: OneShotTurn,
    builtin: &dyn ProviderSource,
) -> anyhow::Result<TurnSummary> {
    let provider = provider_for(graph, &choice.provider_id, builtin).await?;
    let health = tokio::time::timeout(HEALTH_TIMEOUT, provider.health())
        .await
        .unwrap_or_else(|_| {
            nexus_claude::agent::ProviderHealth::unavailable(ProviderError::Timeout {
                after_ms: HEALTH_TIMEOUT.as_millis() as u64,
            })
        });
    if health.status == HealthStatus::Unavailable {
        return Err(anyhow::Error::new(health.error.unwrap_or(
            ProviderError::EndpointUnreachable {
                detail: "the provider reports itself unavailable".to_string(),
            },
        )));
    }
    let mut spec = SessionSpec::new(turn.cwd);
    spec.model = choice.model.clone();
    spec.system_prompt = Some(SystemPromptSpec {
        text: turn.system_prompt,
        mode: SystemPromptMode::Replace,
    });
    spec.policy = ToolPolicy::new(PolicyMode::Trust);
    spec.max_turns = Some(1);
    spec.env.inherit = child_env_inherit_names();
    let session = provider.open(spec).await.map_err(anyhow::Error::new)?;
    let outcome =
        nexus_claude::agent::run_turn(session.as_ref(), TurnInput::text(turn.prompt)).await;
    let _ = session.close().await;
    outcome.map_err(anyhow::Error::new)
}

/// The variables a one-shot process inherits besides nexus' base allowlist: the tooling
/// list and the operator's (`CHAT_CHILD_ENV_INHERIT`), never a server secret — the same
/// list every agent session gets ([`crate::chat::manager::child_env_policy`]).
fn child_env_inherit_names() -> Vec<String> {
    use crate::chat::manager::{CHILD_ENV_INHERIT_VAR, CHILD_ENV_TOOLING, SERVER_ONLY_SECRETS};
    let operator = std::env::var(CHILD_ENV_INHERIT_VAR).unwrap_or_default();
    CHILD_ENV_TOOLING
        .iter()
        .map(|name| (*name).to_string())
        .chain(
            operator
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty() && !SERVER_ONLY_SECRETS.contains(name))
                .map(str::to_string),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::agent_runtime::fake::FakeProvider;
    use crate::chat::provider::settings::{
        ConsentRecord, InstanceRecord, RoleAssignments, RoleTarget, CONSENT_PREFIX, GLOBAL,
        INSTANCE_PREFIX, ROLES_KEY,
    };
    use crate::neo4j::mock::MockGraphStore;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    async fn stored(graph: &MockGraphStore, id: &str) -> InstanceRecord {
        let record = InstanceRecord {
            id: id.into(),
            kind: "acp".into(),
            label: id.into(),
            origin: format!("acp:{id}"),
            credential_ref: "none".into(),
            default_model: Some(format!("{id}-default")),
            ..Default::default()
        };
        graph
            .put_llm_setting(
                GLOBAL,
                &format!("{INSTANCE_PREFIX}{id}"),
                &serde_json::to_string(&record).unwrap(),
            )
            .await
            .unwrap();
        record
    }

    async fn consented(graph: &MockGraphStore, slug: &str, record: &InstanceRecord) {
        let consent = ConsentRecord {
            provider_id: record.id.clone(),
            origin: record.origin.clone(),
            consented_by: "me@example.com".into(),
            consented_at: chrono::Utc::now().to_rfc3339(),
            credential_ref: Some("none".into()),
            ..Default::default()
        };
        graph
            .put_llm_setting(
                &settings::project_scope(slug),
                &format!("{CONSENT_PREFIX}{}", record.id),
                &serde_json::to_string(&consent).unwrap(),
            )
            .await
            .unwrap();
    }

    async fn executor_rule(graph: &MockGraphStore, scope: &str, provider: &str) {
        let roles = RoleAssignments {
            pilot: None,
            executor: Some(RoleTarget {
                provider: provider.into(),
                model: None,
                alias: None,
            }),
        };
        graph
            .put_llm_setting(scope, ROLES_KEY, &serde_json::to_string(&roles).unwrap())
            .await
            .unwrap();
    }

    /// Nothing configured: the built-in Claude Code, with the model the caller named
    /// riding on it. Claude Code absent and nothing else: a typed `no_provider`.
    #[tokio::test]
    async fn a_one_shot_with_nothing_configured_runs_on_claude_code_or_refuses_typed() {
        let graph = MockGraphStore::new();
        let choice = resolve(&graph, Some("demo"), (None, Some("sonnet")), true)
            .await
            .unwrap();
        assert_eq!(choice.provider_id, resolver::CLAUDE_CODE);
        assert_eq!(choice.model.as_deref(), Some("sonnet"));
        assert_eq!(choice.routed_by, RoutedBy::BuiltinClaudeCode);

        let err = resolve(&graph, Some("demo"), (None, None), false)
            .await
            .expect_err("no CLI, nothing else: refused");
        let typed = err
            .downcast_ref::<ResolveError>()
            .expect("the typed refusal");
        assert_eq!(typed.code(), "no_provider");
    }

    /// The project's executor rule names an allowed instance: the one-shot runs there,
    /// on the instance's default model. The same rule on an instance the project never
    /// consented to is skipped with the reason recorded: an executor falls back, it does
    /// not fail — and does not run on the unconsented instance.
    #[tokio::test]
    async fn a_one_shot_follows_the_projects_executor_rule_and_records_a_skipped_one() {
        let graph = MockGraphStore::new();
        let acp = stored(&graph, "acp").await;
        consented(&graph, "demo", &acp).await;
        executor_rule(&graph, &settings::project_scope("demo"), "acp").await;
        let choice = resolve(&graph, Some("demo"), (None, None), true)
            .await
            .unwrap();
        assert_eq!(choice.provider_id, "acp");
        assert_eq!(choice.routed_by, RoutedBy::ProjectRule);
        assert_eq!(choice.model, None, "the instance's default model");

        // Another project, same rule, no consent there.
        executor_rule(&graph, &settings::project_scope("other"), "acp").await;
        let choice = resolve(&graph, Some("other"), (None, None), true)
            .await
            .unwrap();
        assert_eq!(choice.provider_id, resolver::CLAUDE_CODE);
        assert_eq!(choice.routed_by, RoutedBy::Fallback);
        assert_eq!(
            choice.fallback_reason.as_deref(),
            Some("endpoint_not_allowed")
        );
    }

    /// An explicit request (the router's applied pick, or `FEATURE_GRAPH_MODEL` with a
    /// provider) wins over the rules and is never substituted: unconsented, it is a
    /// typed refusal, not a fallback.
    #[tokio::test]
    async fn an_explicit_request_wins_over_the_rules_and_is_never_substituted() {
        let graph = MockGraphStore::new();
        let acp = stored(&graph, "acp").await;
        consented(&graph, "demo", &acp).await;
        let other = stored(&graph, "other").await;
        consented(&graph, "demo", &other).await;
        executor_rule(&graph, &settings::project_scope("demo"), "other").await;
        let pick = Candidate::new("acp", Some("acp-fast".into()));
        let choice = resolve_candidate(&graph, Some("demo"), Some(&pick), true)
            .await
            .unwrap();
        assert_eq!(choice.provider_id, "acp");
        assert_eq!(choice.model.as_deref(), Some("acp-fast"));
        assert_eq!(choice.routed_by, RoutedBy::Request);

        let err = resolve_candidate(&graph, Some("nowhere"), Some(&pick), true)
            .await
            .expect_err("explicit and unconsented: refused");
        assert_eq!(
            err.downcast_ref::<ResolveError>().map(ResolveError::code),
            Some("endpoint_not_allowed")
        );
    }

    /// The turn itself, on the scripted provider: one session opened in trust with one
    /// turn and the system prompt, the text of its `done` event returned, the session
    /// closed. An instance whose health fails is the typed error, and nothing opens.
    #[tokio::test]
    async fn a_one_shot_opens_one_session_runs_one_turn_and_closes_it() {
        use nexus_claude::agent::{AgentEvent, Cost, CostBasis, ProviderHealth, StopReason};
        let graph = MockGraphStore::new();
        let fake = FakeProvider::new();
        let choice = OneShotChoice {
            provider_id: resolver::CLAUDE_CODE.into(),
            model: Some("sonnet".into()),
            routed_by: RoutedBy::BuiltinClaudeCode,
            fallback_reason: None,
        };
        let pusher = fake.clone();
        tokio::spawn(async move {
            while pusher.state.turns_started.lock().unwrap().is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            pusher.state.push(AgentEvent::Done {
                stop_reason: StopReason::Completed,
                subtype: Some("success".into()),
                is_error: false,
                result_text: Some("[{\"name\":\"auth\"}]".into()),
                usage: Default::default(),
                cost: Cost {
                    usd: Some(0.01),
                    basis: CostBasis::Reported,
                },
                duration_ms: 5,
                duration_api_ms: None,
                num_turns: 1,
                model: Some("sonnet".into()),
                provider_session_id: Some("p-1".into()),
                structured_output: None,
                error: None,
            });
        });
        let summary = run(
            &graph,
            &choice,
            OneShotTurn {
                cwd: std::env::temp_dir(),
                system_prompt: "Respond only with JSON.".into(),
                prompt: "propose".into(),
            },
            &fake,
        )
        .await
        .unwrap_or_else(|e| panic!("the turn runs: {e:#}"));
        assert_eq!(summary.text, "[{\"name\":\"auth\"}]");
        assert!(!summary.is_error);
        {
            let specs = fake.state.opened_specs.lock().unwrap();
            assert_eq!(specs.len(), 1);
            assert_eq!(specs[0].model.as_deref(), Some("sonnet"));
            assert_eq!(specs[0].max_turns, Some(1));
            assert_eq!(specs[0].policy.mode, PolicyMode::Trust);
            assert_eq!(
                specs[0].system_prompt.as_ref().map(|p| p.text.as_str()),
                Some("Respond only with JSON.")
            );
        }
        assert_eq!(fake.state.turns_started.lock().unwrap()[0], "propose");
        assert!(fake.state.closed.load(std::sync::atomic::Ordering::SeqCst));

        // Health fails: typed, nothing opened.
        let down = FakeProvider::new();
        *down.health.lock().unwrap() =
            Some(ProviderHealth::unavailable(ProviderError::AuthRequired {
                login_hint: Some("claude login".into()),
            }));
        let err = run(
            &graph,
            &choice,
            OneShotTurn {
                cwd: std::env::temp_dir(),
                system_prompt: String::new(),
                prompt: "propose".into(),
            },
            &down,
        )
        .await
        .expect_err("unavailable: refused");
        assert!(matches!(
            err.downcast_ref::<ProviderError>(),
            Some(ProviderError::AuthRequired { .. })
        ));
        assert!(down.state.opened_specs.lock().unwrap().is_empty());

        // An id nobody registered is the typed refusal, not a bare error.
        let code = match provider_for(&graph, "ghost", &fake).await {
            Ok(_) => panic!("an unknown id does not build"),
            Err(err) => err.downcast_ref::<ResolveError>().map(ResolveError::code),
        };
        assert_eq!(code, Some("provider_unknown"));
    }
}
