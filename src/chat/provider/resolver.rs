//! Provider and model resolution (decisions A1, A15, A16, A18).
//!
//! Pure logic over an abstract [`InstanceCatalog`]: the registry of provider
//! instances lives in nexus, the backend only decides which instance serves a
//! given request.
//!
//! Precedence (A16): existing session (frozen) > explicit request > task >
//! persona > run > project rule > global rule > configured default >
//! `claude-code` when usable > `no_provider`.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Identifier of the built-in Claude Code instance.
pub const CLAUDE_CODE: &str = "claude-code";

/// Role of the session being opened (A15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Session opened by a human. Never falls back automatically.
    Pilot,
    /// Runner, delegation, protocol or one-shot session. May fall back.
    Executor,
}

impl Role {
    /// Stable serialised name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pilot => "pilot",
            Self::Executor => "executor",
        }
    }
}

/// Which precedence level produced the choice. Persisted with the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoutedBy {
    /// The existing session's provider (frozen).
    #[serde(rename = "session")]
    Session,
    /// Named by the request.
    #[serde(rename = "request")]
    Request,
    /// Named by the task.
    #[serde(rename = "task")]
    Task,
    /// Named by the persona.
    #[serde(rename = "persona")]
    Persona,
    /// Named by the run.
    #[serde(rename = "run")]
    Run,
    /// Project routing rule.
    #[serde(rename = "project_rule")]
    ProjectRule,
    /// Global routing rule.
    #[serde(rename = "global_rule")]
    GlobalRule,
    /// Configured default provider.
    #[serde(rename = "default")]
    ConfiguredDefault,
    /// Nothing configured: the built-in Claude Code instance.
    #[serde(rename = "claude_code")]
    BuiltinClaudeCode,
    /// A higher level was skipped (unhealthy or not allowed) for an executor.
    #[serde(rename = "fallback")]
    Fallback,
}

impl RoutedBy {
    /// Stable serialised name (identical to the serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Request => "request",
            Self::Task => "task",
            Self::Persona => "persona",
            Self::Run => "run",
            Self::ProjectRule => "project_rule",
            Self::GlobalRule => "global_rule",
            Self::ConfiguredDefault => "default",
            Self::BuiltinClaudeCode => "claude_code",
            Self::Fallback => "fallback",
        }
    }
}

/// The resolved provider instance and model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderChoice {
    /// Instance identifier (registry key).
    pub provider_id: String,
    /// Model, when the winning level named one.
    pub model: Option<String>,
    /// Level that produced the choice.
    pub routed_by: RoutedBy,
}

/// A provider (and optional model) proposed by one precedence level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// Instance identifier (registry key).
    pub provider_id: String,
    /// Model, when the level names one.
    pub model: Option<String>,
}

impl Candidate {
    /// A candidate naming an instance and, optionally, a model.
    pub fn new(provider_id: impl Into<String>, model: Option<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model,
        }
    }
}

/// Everything the resolver looks at, in precedence order (A16).
#[derive(Debug, Clone)]
pub struct ResolveInput<'a> {
    /// Provider of the existing session, frozen: a resume never re-resolves.
    pub session: Option<&'a Candidate>,
    /// Explicit choice of the request.
    pub request: Option<Candidate>,
    /// Choice carried by the task.
    pub task: Option<Candidate>,
    /// Choice carried by the persona.
    pub persona: Option<Candidate>,
    /// Choice carried by the run.
    pub run: Option<Candidate>,
    /// Project routing rule.
    pub project_rule: Option<Candidate>,
    /// Global routing rule.
    pub global_rule: Option<Candidate>,
    /// Configured default provider.
    pub configured_default: Option<Candidate>,
    /// Role of the session being opened.
    pub role: Role,
}

impl ResolveInput<'_> {
    /// An input with no level set.
    pub fn empty(role: Role) -> Self {
        Self {
            session: None,
            request: None,
            task: None,
            persona: None,
            run: None,
            project_rule: None,
            global_rule: None,
            configured_default: None,
            role,
        }
    }
}

/// View of the provider instances, for one fixed project (chosen by the implementer).
pub trait InstanceCatalog {
    /// Whether the instance is registered.
    fn exists(&self, provider_id: &str) -> bool;
    /// Whether the instance can open sessions right now.
    fn is_healthy(&self, provider_id: &str) -> bool;
    /// Whether the project consented to this instance (A28).
    fn is_allowed_for_project(&self, provider_id: &str) -> bool;
}

/// Why no provider could be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The request names another provider than the session's (HTTP 409).
    ProviderConflict {
        /// Provider of the existing session.
        session: String,
        /// Provider named by the request.
        requested: String,
    },
    /// The instance is not registered.
    UnknownProvider(String),
    /// The project did not consent to this instance.
    NotAllowed(String),
    /// The instance cannot open sessions and no fallback applies.
    Unavailable {
        /// Instance identifier.
        provider_id: String,
        /// Role of the session being opened.
        role: Role,
    },
    /// Nothing is configured and Claude Code is not usable.
    NoProvider,
}

impl ResolveError {
    /// Stable error code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::ProviderConflict { .. } => "provider_conflict",
            Self::UnknownProvider(_) => "provider_unknown",
            Self::NotAllowed(_) => "endpoint_not_allowed",
            Self::Unavailable { .. } => "provider_unavailable",
            Self::NoProvider => "no_provider",
        }
    }
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProviderConflict { session, requested } => write!(
                f,
                "session is bound to provider '{session}', request names '{requested}'"
            ),
            Self::UnknownProvider(id) => write!(f, "unknown provider '{id}'"),
            Self::NotAllowed(id) => {
                write!(f, "provider '{id}' is not allowed for this project")
            }
            Self::Unavailable { provider_id, role } => write!(
                f,
                "provider '{provider_id}' is unavailable for role {}",
                role.as_str()
            ),
            Self::NoProvider => f.write_str("no provider is configured"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Checks one candidate against the catalog: registered, allowed, healthy.
fn check(
    candidate: &Candidate,
    role: Role,
    catalog: &dyn InstanceCatalog,
) -> Result<(), ResolveError> {
    let id = candidate.provider_id.as_str();
    if !catalog.exists(id) {
        return Err(ResolveError::UnknownProvider(id.to_string()));
    }
    if !catalog.is_allowed_for_project(id) {
        return Err(ResolveError::NotAllowed(id.to_string()));
    }
    if !catalog.is_healthy(id) {
        return Err(ResolveError::Unavailable {
            provider_id: id.to_string(),
            role,
        });
    }
    Ok(())
}

/// Resolves the provider instance and model for one request.
///
/// - An existing session is frozen: its provider wins without any check, and a
///   request naming another provider is a [`ResolveError::ProviderConflict`].
/// - An explicit choice (request, task, persona) is never substituted.
/// - A pilot never falls back, at any level.
/// - An executor skips a non-explicit level that is unhealthy or not allowed;
///   the final choice then carries [`RoutedBy::Fallback`]. An unregistered
///   instance is a configuration error at every level and is never skipped.
pub fn resolve(
    input: &ResolveInput<'_>,
    catalog: &dyn InstanceCatalog,
) -> Result<ProviderChoice, ResolveError> {
    // 1. Existing session: frozen, never re-resolved (even when unhealthy).
    if let Some(session) = input.session {
        if let Some(request) = &input.request {
            if request.provider_id != session.provider_id {
                return Err(ResolveError::ProviderConflict {
                    session: session.provider_id.clone(),
                    requested: request.provider_id.clone(),
                });
            }
        }
        return Ok(ProviderChoice {
            provider_id: session.provider_id.clone(),
            model: session.model.clone(),
            routed_by: RoutedBy::Session,
        });
    }

    // 2. Levels in precedence order; the flag marks explicit choices.
    let levels: [(Option<&Candidate>, RoutedBy, bool); 7] = [
        (input.request.as_ref(), RoutedBy::Request, true),
        (input.task.as_ref(), RoutedBy::Task, true),
        (input.persona.as_ref(), RoutedBy::Persona, true),
        (input.run.as_ref(), RoutedBy::Run, false),
        (input.project_rule.as_ref(), RoutedBy::ProjectRule, false),
        (input.global_rule.as_ref(), RoutedBy::GlobalRule, false),
        (
            input.configured_default.as_ref(),
            RoutedBy::ConfiguredDefault,
            false,
        ),
    ];

    // First level skipped by an executor: reported when the chain is exhausted.
    let mut first_skip: Option<ResolveError> = None;

    for (candidate, routed_by, explicit) in levels {
        let candidate = match candidate {
            Some(candidate) => candidate,
            None => continue,
        };
        match check(candidate, input.role, catalog) {
            Ok(()) => {
                return Ok(ProviderChoice {
                    provider_id: candidate.provider_id.clone(),
                    model: candidate.model.clone(),
                    routed_by: if first_skip.is_some() {
                        RoutedBy::Fallback
                    } else {
                        routed_by
                    },
                });
            }
            Err(error) => {
                let skippable = !explicit
                    && input.role == Role::Executor
                    && !matches!(error, ResolveError::UnknownProvider(_));
                if !skippable {
                    return Err(error);
                }
                if first_skip.is_none() {
                    first_skip = Some(error);
                }
            }
        }
    }

    // 3. Nothing configured (or every level skipped): built-in Claude Code.
    let claude_usable = catalog.exists(CLAUDE_CODE)
        && catalog.is_healthy(CLAUDE_CODE)
        && catalog.is_allowed_for_project(CLAUDE_CODE);
    if claude_usable {
        return Ok(ProviderChoice {
            provider_id: CLAUDE_CODE.to_string(),
            model: None,
            routed_by: if first_skip.is_some() {
                RoutedBy::Fallback
            } else {
                RoutedBy::BuiltinClaudeCode
            },
        });
    }
    match first_skip {
        // Fallback chain exhausted: fail with the reason of the first skipped level.
        Some(error) => Err(error),
        None => Err(ResolveError::NoProvider),
    }
}

/// Catalog used until the instance registry is wired in: the built-in
/// `claude-code` instance is the only one that exists, and it is always usable
/// (its health is reported at open time by the typed opening errors).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinCatalog;

impl InstanceCatalog for BuiltinCatalog {
    fn exists(&self, provider_id: &str) -> bool {
        provider_id == CLAUDE_CODE
    }
    fn is_healthy(&self, _provider_id: &str) -> bool {
        true
    }
    fn is_allowed_for_project(&self, _provider_id: &str) -> bool {
        true
    }
}

/// Resolves the provider of a pilot session being opened or resumed.
///
/// - `stored` is the provider persisted on an existing session (`None` for a
///   session written before the harness existed, which is `claude-code`).
/// - `requested` is the explicit `provider` of the request.
pub fn resolve_for_open(
    stored: Option<&str>,
    is_existing_session: bool,
    requested: Option<&str>,
    catalog: &dyn InstanceCatalog,
) -> Result<ProviderChoice, ResolveError> {
    let session = is_existing_session.then(|| Candidate::new(stored.unwrap_or(CLAUDE_CODE), None));
    let mut input = ResolveInput::empty(Role::Pilot);
    input.session = session.as_ref();
    input.request = requested
        .filter(|id| !id.is_empty())
        .map(|id| Candidate::new(id, None));
    resolve(&input, catalog)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table-driven catalog: `(id, healthy, allowed)`.
    struct FakeCatalog {
        rows: Vec<(&'static str, bool, bool)>,
    }

    impl FakeCatalog {
        fn new(rows: &[(&'static str, bool, bool)]) -> Self {
            Self {
                rows: rows.to_vec(),
            }
        }

        fn row(&self, id: &str) -> Option<(bool, bool)> {
            self.rows
                .iter()
                .find(|row| row.0 == id)
                .map(|row| (row.1, row.2))
        }
    }

    impl InstanceCatalog for FakeCatalog {
        fn exists(&self, provider_id: &str) -> bool {
            self.row(provider_id).is_some()
        }

        fn is_healthy(&self, provider_id: &str) -> bool {
            self.row(provider_id).map(|row| row.0).unwrap_or(false)
        }

        fn is_allowed_for_project(&self, provider_id: &str) -> bool {
            self.row(provider_id).map(|row| row.1).unwrap_or(false)
        }
    }

    const LEVEL_IDS: [&str; 7] = [
        "p-request",
        "p-task",
        "p-persona",
        "p-run",
        "p-project",
        "p-global",
        "p-default",
    ];

    fn cand(id: &str) -> Candidate {
        Candidate::new(id, Some(format!("{id}-model")))
    }

    /// Catalog where every level instance and Claude Code are healthy and allowed.
    fn all_good() -> FakeCatalog {
        let mut rows: Vec<(&'static str, bool, bool)> =
            LEVEL_IDS.iter().map(|id| (*id, true, true)).collect();
        rows.push((CLAUDE_CODE, true, true));
        FakeCatalog { rows }
    }

    /// Input where every level from `start` (0 = request) downwards is set.
    fn filled_from(start: usize, role: Role) -> ResolveInput<'static> {
        let pick = |index: usize| {
            if index >= start {
                Some(cand(LEVEL_IDS[index]))
            } else {
                None
            }
        };
        ResolveInput {
            session: None,
            request: pick(0),
            task: pick(1),
            persona: pick(2),
            run: pick(3),
            project_rule: pick(4),
            global_rule: pick(5),
            configured_default: pick(6),
            role,
        }
    }

    fn assert_level(start: usize, expected: RoutedBy) {
        for role in [Role::Pilot, Role::Executor] {
            let choice = resolve(&filled_from(start, role), &all_good()).unwrap();
            assert_eq!(choice.provider_id, LEVEL_IDS[start]);
            assert_eq!(choice.model, Some(format!("{}-model", LEVEL_IDS[start])));
            assert_eq!(choice.routed_by, expected);
        }
    }

    #[test]
    fn session_wins_over_every_level_and_is_never_re_resolved() {
        // The session's instance is unhealthy, not allowed, even unregistered:
        // a resume never re-resolves.
        let session = Candidate::new("gone", Some("m1".to_string()));
        let catalog = all_good();
        for role in [Role::Pilot, Role::Executor] {
            let mut input = filled_from(1, role);
            input.session = Some(&session);
            let choice = resolve(&input, &catalog).unwrap();
            assert_eq!(choice.provider_id, "gone");
            assert_eq!(choice.model, Some("m1".to_string()));
            assert_eq!(choice.routed_by, RoutedBy::Session);
        }
    }

    #[test]
    fn session_and_request_naming_another_provider_is_a_conflict() {
        let session = Candidate::new("p-task", None);
        let mut input = ResolveInput::empty(Role::Pilot);
        input.session = Some(&session);
        input.request = Some(cand("p-request"));
        let error = resolve(&input, &all_good()).unwrap_err();
        assert_eq!(
            error,
            ResolveError::ProviderConflict {
                session: "p-task".to_string(),
                requested: "p-request".to_string(),
            }
        );
        assert_eq!(error.code(), "provider_conflict");
    }

    #[test]
    fn session_and_request_with_same_provider_other_model_is_not_a_conflict() {
        let session = Candidate::new("p-request", Some("session-model".to_string()));
        let mut input = ResolveInput::empty(Role::Executor);
        input.session = Some(&session);
        input.request = Some(Candidate::new("p-request", Some("other-model".to_string())));
        let choice = resolve(&input, &all_good()).unwrap();
        assert_eq!(choice.provider_id, "p-request");
        assert_eq!(choice.model, Some("session-model".to_string()));
        assert_eq!(choice.routed_by, RoutedBy::Session);
    }

    #[test]
    fn level_request_wins_first() {
        assert_level(0, RoutedBy::Request);
    }

    #[test]
    fn level_task_wins_when_no_request() {
        assert_level(1, RoutedBy::Task);
    }

    #[test]
    fn level_persona_wins_when_no_task() {
        assert_level(2, RoutedBy::Persona);
    }

    #[test]
    fn level_run_wins_when_no_persona() {
        assert_level(3, RoutedBy::Run);
    }

    #[test]
    fn level_project_rule_wins_when_no_run() {
        assert_level(4, RoutedBy::ProjectRule);
    }

    #[test]
    fn level_global_rule_wins_when_no_project_rule() {
        assert_level(5, RoutedBy::GlobalRule);
    }

    #[test]
    fn level_configured_default_wins_when_no_rule() {
        assert_level(6, RoutedBy::ConfiguredDefault);
    }

    #[test]
    fn builtin_claude_code_when_nothing_is_configured() {
        for role in [Role::Pilot, Role::Executor] {
            let choice = resolve(&ResolveInput::empty(role), &all_good()).unwrap();
            assert_eq!(choice.provider_id, CLAUDE_CODE);
            assert_eq!(choice.model, None);
            assert_eq!(choice.routed_by, RoutedBy::BuiltinClaudeCode);
        }
    }

    #[test]
    fn explicit_choice_is_never_substituted() {
        // Lower levels and Claude Code are fine; the explicit level is not.
        let catalog = FakeCatalog::new(&[
            ("sick", false, true),
            ("forbidden", true, false),
            ("p-run", true, true),
            ("p-default", true, true),
            (CLAUDE_CODE, true, true),
        ]);
        for role in [Role::Pilot, Role::Executor] {
            for level in 0..3usize {
                let cases = [
                    ("ghost", ResolveError::UnknownProvider("ghost".to_string())),
                    (
                        "forbidden",
                        ResolveError::NotAllowed("forbidden".to_string()),
                    ),
                    (
                        "sick",
                        ResolveError::Unavailable {
                            provider_id: "sick".to_string(),
                            role,
                        },
                    ),
                ];
                for (id, expected) in cases {
                    let mut input = ResolveInput::empty(role);
                    input.run = Some(cand("p-run"));
                    input.configured_default = Some(cand("p-default"));
                    match level {
                        0 => input.request = Some(cand(id)),
                        1 => input.task = Some(cand(id)),
                        _ => input.persona = Some(cand(id)),
                    }
                    assert_eq!(resolve(&input, &catalog).unwrap_err(), expected);
                }
            }
        }
    }

    #[test]
    fn pilot_never_falls_back_at_any_level() {
        let catalog = FakeCatalog::new(&[
            ("sick", false, true),
            ("forbidden", true, false),
            ("p-default", true, true),
            (CLAUDE_CODE, true, true),
        ]);
        for level in 0..3usize {
            let mut input = ResolveInput::empty(Role::Pilot);
            input.configured_default = Some(cand("p-default"));
            match level {
                0 => input.run = Some(cand("sick")),
                1 => input.project_rule = Some(cand("sick")),
                _ => input.global_rule = Some(cand("sick")),
            }
            let error = resolve(&input, &catalog).unwrap_err();
            assert_eq!(
                error,
                ResolveError::Unavailable {
                    provider_id: "sick".to_string(),
                    role: Role::Pilot,
                }
            );
            assert_eq!(error.code(), "provider_unavailable");
        }
        // Unhealthy configured default: no silent switch to Claude Code either.
        let mut input = ResolveInput::empty(Role::Pilot);
        input.configured_default = Some(cand("sick"));
        assert!(matches!(
            resolve(&input, &catalog),
            Err(ResolveError::Unavailable { .. })
        ));
        // Not allowed is an error too, not a skip.
        let mut input = ResolveInput::empty(Role::Pilot);
        input.project_rule = Some(cand("forbidden"));
        input.configured_default = Some(cand("p-default"));
        assert_eq!(
            resolve(&input, &catalog).unwrap_err(),
            ResolveError::NotAllowed("forbidden".to_string())
        );
    }

    #[test]
    fn executor_falls_back_to_the_next_level_and_records_it() {
        let catalog = FakeCatalog::new(&[
            ("sick", false, true),
            ("p-global", true, true),
            (CLAUDE_CODE, true, true),
        ]);
        let mut input = ResolveInput::empty(Role::Executor);
        input.run = Some(cand("sick"));
        input.global_rule = Some(cand("p-global"));
        let choice = resolve(&input, &catalog).unwrap();
        assert_eq!(choice.provider_id, "p-global");
        assert_eq!(choice.model, Some("p-global-model".to_string()));
        assert_eq!(choice.routed_by, RoutedBy::Fallback);
    }

    #[test]
    fn executor_fallback_skips_a_not_allowed_instance() {
        let catalog = FakeCatalog::new(&[
            ("sick", false, true),
            ("forbidden", true, false),
            ("p-default", true, true),
            (CLAUDE_CODE, true, true),
        ]);
        let mut input = ResolveInput::empty(Role::Executor);
        input.run = Some(cand("sick"));
        input.project_rule = Some(cand("forbidden"));
        input.configured_default = Some(cand("p-default"));
        let choice = resolve(&input, &catalog).unwrap();
        assert_eq!(choice.provider_id, "p-default");
        assert_eq!(choice.routed_by, RoutedBy::Fallback);
    }

    #[test]
    fn executor_fallback_reaches_claude_code_and_is_still_recorded_as_fallback() {
        let catalog = FakeCatalog::new(&[("sick", false, true), (CLAUDE_CODE, true, true)]);
        let mut input = ResolveInput::empty(Role::Executor);
        input.configured_default = Some(cand("sick"));
        let choice = resolve(&input, &catalog).unwrap();
        assert_eq!(choice.provider_id, CLAUDE_CODE);
        assert_eq!(choice.routed_by, RoutedBy::Fallback);
    }

    #[test]
    fn executor_with_an_exhausted_chain_fails_with_the_first_skipped_reason() {
        // Claude Code exists but the project did not allow it: never used.
        let catalog = FakeCatalog::new(&[
            ("sick", false, true),
            ("forbidden", true, false),
            (CLAUDE_CODE, true, false),
        ]);
        let mut input = ResolveInput::empty(Role::Executor);
        input.run = Some(cand("sick"));
        input.global_rule = Some(cand("forbidden"));
        assert_eq!(
            resolve(&input, &catalog).unwrap_err(),
            ResolveError::Unavailable {
                provider_id: "sick".to_string(),
                role: Role::Executor,
            }
        );
    }

    #[test]
    fn executor_does_not_skip_an_unregistered_rule_target() {
        let catalog = FakeCatalog::new(&[("p-default", true, true), (CLAUDE_CODE, true, true)]);
        let mut input = ResolveInput::empty(Role::Executor);
        input.project_rule = Some(cand("ghost"));
        input.configured_default = Some(cand("p-default"));
        let error = resolve(&input, &catalog).unwrap_err();
        assert_eq!(error, ResolveError::UnknownProvider("ghost".to_string()));
        assert_eq!(error.code(), "provider_unknown");
    }

    #[test]
    fn no_provider_when_nothing_is_configured_and_claude_code_is_unusable() {
        let catalogs = [
            FakeCatalog::new(&[]),
            FakeCatalog::new(&[(CLAUDE_CODE, false, true)]),
            FakeCatalog::new(&[(CLAUDE_CODE, true, false)]),
        ];
        for catalog in &catalogs {
            for role in [Role::Pilot, Role::Executor] {
                let error = resolve(&ResolveInput::empty(role), catalog).unwrap_err();
                assert_eq!(error, ResolveError::NoProvider);
                assert_eq!(error.code(), "no_provider");
            }
        }
    }

    #[test]
    fn routed_by_and_role_names_are_stable_and_match_serde() {
        let table = [
            (RoutedBy::Session, "session"),
            (RoutedBy::Request, "request"),
            (RoutedBy::Task, "task"),
            (RoutedBy::Persona, "persona"),
            (RoutedBy::Run, "run"),
            (RoutedBy::ProjectRule, "project_rule"),
            (RoutedBy::GlobalRule, "global_rule"),
            (RoutedBy::ConfiguredDefault, "default"),
            (RoutedBy::BuiltinClaudeCode, "claude_code"),
            (RoutedBy::Fallback, "fallback"),
        ];
        for (routed_by, name) in table {
            assert_eq!(routed_by.as_str(), name);
            assert_eq!(
                serde_json::to_value(routed_by).unwrap(),
                serde_json::json!(name)
            );
            let back: RoutedBy = serde_json::from_value(serde_json::json!(name)).unwrap();
            assert_eq!(back, routed_by);
        }
        for (role, name) in [(Role::Pilot, "pilot"), (Role::Executor, "executor")] {
            assert_eq!(role.as_str(), name);
            assert_eq!(serde_json::to_value(role).unwrap(), serde_json::json!(name));
        }
    }

    #[test]
    fn error_codes_and_messages_are_stable() {
        let not_allowed = ResolveError::NotAllowed("x".to_string());
        assert_eq!(not_allowed.code(), "endpoint_not_allowed");
        assert_eq!(
            not_allowed.to_string(),
            "provider 'x' is not allowed for this project"
        );
        let unavailable = ResolveError::Unavailable {
            provider_id: "x".to_string(),
            role: Role::Executor,
        };
        assert_eq!(
            unavailable.to_string(),
            "provider 'x' is unavailable for role executor"
        );
        assert_eq!(
            ResolveError::NoProvider.to_string(),
            "no provider is configured"
        );
    }

    #[test]
    fn open_without_any_choice_is_claude_code() {
        let c = resolve_for_open(None, false, None, &BuiltinCatalog).unwrap();
        assert_eq!(c.provider_id, CLAUDE_CODE);
        assert_eq!(c.routed_by, RoutedBy::BuiltinClaudeCode);
    }

    #[test]
    fn open_names_an_unregistered_provider_is_unknown() {
        let e = resolve_for_open(None, false, Some("deepseek"), &BuiltinCatalog).unwrap_err();
        assert_eq!(e.code(), "provider_unknown");
    }

    #[test]
    fn an_existing_legacy_session_is_claude_code_and_frozen() {
        let c = resolve_for_open(None, true, None, &BuiltinCatalog).unwrap();
        assert_eq!(
            (c.provider_id.as_str(), c.routed_by),
            (CLAUDE_CODE, RoutedBy::Session)
        );
        let e = resolve_for_open(None, true, Some("deepseek"), &BuiltinCatalog).unwrap_err();
        assert_eq!(e.code(), "provider_conflict");
    }

    #[test]
    fn naming_claude_code_explicitly_is_a_request_choice() {
        let c = resolve_for_open(None, false, Some(CLAUDE_CODE), &BuiltinCatalog).unwrap();
        assert_eq!(c.routed_by, RoutedBy::Request);
    }
}
