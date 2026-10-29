//! The instance catalog the resolver reads, built from what is stored
//! (instances, the consent of the project being served, roles, aliases).
//!
//! The resolver stays pure ([`super::resolver::InstanceCatalog`] is a sync
//! trait over facts); this module loads those facts once per opening, so a
//! decision is taken on one consistent view.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use std::collections::{HashMap, HashSet};

use super::resolver::{Candidate, InstanceCatalog, ResolveInput, Role, CLAUDE_CODE};
use super::settings::{
    ConsentRecord, InstanceRecord, ModelAlias, ModelPolicy, RoleAssignments, RoleTarget,
};

/// Facts about the instances for ONE consent scope: a project, or the sessions
/// outside any project.
#[derive(Debug, Clone, Default)]
pub struct StoreCatalog {
    instances: HashMap<String, InstanceRecord>,
    consented: HashSet<String>,
    unhealthy: HashSet<String>,
    /// The Claude Code CLI is NOT on the server (B2): the built-in instance is no default.
    claude_cli_missing: bool,
}

impl StoreCatalog {
    /// Builds the catalog. A consent counts only while it is tied to the
    /// instance's CURRENT origin (A28). `consents` are those of the scope the
    /// opening is in (`store::consents_for`): the project's, or, outside any
    /// project, the ones a person gave for the sessions without one (B3).
    pub fn new(instances: Vec<InstanceRecord>, consents: &[ConsentRecord]) -> Self {
        let consented = instances
            .iter()
            .filter(|i| {
                consents
                    .iter()
                    .any(|c| super::settings::consent_holds(c, i))
            })
            .map(|i| i.id.clone())
            .collect();
        Self {
            instances: instances.into_iter().map(|i| (i.id.clone(), i)).collect(),
            consented,
            unhealthy: HashSet::new(),
            claude_cli_missing: false,
        }
    }

    /// Whether the Claude Code CLI is on the server (`ChatManager::claude_code_usable`).
    pub fn with_claude_cli(mut self, present: bool) -> Self {
        self.claude_cli_missing = !present;
        self
    }

    /// Marks instances a health check found unusable.
    pub fn with_unhealthy(mut self, ids: impl IntoIterator<Item = String>) -> Self {
        self.unhealthy.extend(ids);
        self
    }

    /// The stored record of an instance.
    pub fn instance(&self, id: &str) -> Option<&InstanceRecord> {
        self.instances.get(id)
    }
}

impl InstanceCatalog for StoreCatalog {
    fn exists(&self, provider_id: &str) -> bool {
        provider_id == CLAUDE_CODE || self.instances.contains_key(provider_id)
    }

    fn is_healthy(&self, provider_id: &str) -> bool {
        !self.unhealthy.contains(provider_id)
    }

    fn is_allowed_for_project(&self, provider_id: &str) -> bool {
        provider_id == CLAUDE_CODE
            || self.consented.contains(provider_id)
            || self
                .instances
                .get(provider_id)
                .is_some_and(|i| i.auto_consent)
    }

    fn claude_code_installed(&self) -> bool {
        !self.claude_cli_missing
    }

    fn instance_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.instances.keys().cloned().collect();
        ids.sort();
        ids
    }
}

/// Turns a role target into a candidate: an alias goes through the alias
/// table (an alias that is not defined yields nothing, never a guess).
pub fn candidate_of(target: &RoleTarget, aliases: &[ModelAlias]) -> Option<Candidate> {
    match &target.alias {
        Some(alias) => aliases
            .iter()
            .find(|a| &a.alias == alias)
            .map(|a| Candidate::new(a.provider.clone(), Some(a.model.clone()))),
        None => Some(Candidate::new(
            target.provider.clone(),
            target.model.clone(),
        )),
    }
}

/// What the model policy (A19) says for a session: the rule that matched and
/// the candidate it points at. `off` says nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyPick {
    /// Rule role that matched (`chat`, `runner.simple`, ...).
    pub rule: String,
    /// Where that rule's alias leads.
    pub candidate: Candidate,
    /// `enforce` (applied) or `shadow` (recorded, never applied).
    pub enforced: bool,
}

/// Rule role of a session: a pilot is `chat`; an executor is `runner.<class>`
/// when its task class is one the policy knows.
pub fn rule_role(role: Role, task_class: Option<&str>) -> Option<String> {
    match role {
        Role::Pilot => Some("chat".to_string()),
        Role::Executor => task_class
            .filter(|c| matches!(*c, "simple" | "complex" | "creative" | "retry"))
            .map(|c| format!("runner.{c}")),
    }
}

/// Reads the policy for one session. An alias the rule names that is not
/// defined matches nothing (never a guess); `off` says nothing.
pub fn policy_pick(
    policy: &ModelPolicy,
    role: Role,
    task_class: Option<&str>,
    aliases: &[ModelAlias],
) -> Option<PolicyPick> {
    let enforced = match policy.mode.as_str() {
        "enforce" => true,
        "shadow" => false,
        _ => return None,
    };
    let rule = rule_role(role, task_class)?;
    let alias = policy.rules.get(&rule)?;
    let target = aliases.iter().find(|a| &a.alias == alias)?;
    Some(PolicyPick {
        rule,
        candidate: Candidate::new(target.provider.clone(), Some(target.model.clone())),
        enforced,
    })
}

/// The candidate an alias names, through the alias table. Empty or undefined
/// alias: no level. Shared by the task level and the persona level (A16), so a
/// persona preference is matched exactly like a task alias.
pub fn alias_candidate(alias: Option<&str>, aliases: &[ModelAlias]) -> Option<Candidate> {
    alias
        .filter(|a| !a.is_empty())
        .and_then(|alias| aliases.iter().find(|a| a.alias == alias))
        .map(|a| Candidate::new(a.provider.clone(), Some(a.model.clone())))
}

/// Fills the resolver input from the stored roles: the project's role is the
/// project rule, the global one the global rule (A16). An explicit request is
/// the caller's.
#[allow(clippy::too_many_arguments)]
pub fn resolve_input<'a>(
    role: Role,
    request_provider: Option<&str>,
    task_alias: Option<&str>,
    run: Option<(&str, Option<&str>)>,
    global: &RoleAssignments,
    project: &RoleAssignments,
    aliases: &[ModelAlias],
) -> ResolveInput<'a> {
    let pick = |roles: &RoleAssignments| {
        let target = match role {
            Role::Pilot => roles.pilot.as_ref(),
            Role::Executor => roles.executor.as_ref(),
        };
        target.and_then(|t| candidate_of(t, aliases))
    };
    let mut input = ResolveInput::empty(role);
    input.request = request_provider
        .filter(|p| !p.is_empty())
        .map(|p| Candidate::new(p, None));
    // The task's alias goes through the alias table; an undefined one is no level.
    input.task = alias_candidate(task_alias, aliases);
    input.run = run
        .filter(|(p, _)| !p.is_empty())
        .map(|(p, m)| Candidate::new(p, m.map(str::to_string)));
    input.project_rule = pick(project);
    input.global_rule = pick(global);
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::resolver::{resolve, RoutedBy};
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn instance(id: &str, origin: &str) -> InstanceRecord {
        InstanceRecord {
            id: id.into(),
            kind: "openai_compatible".into(),
            preset: None,
            label: id.into(),
            base_url: format!("{origin}/v1"),
            origin: origin.into(),
            default_model: Some("m".into()),
            cost_source: "unknown".into(),
            credential_ref: "none".into(),
            ..Default::default()
        }
    }

    fn consent(id: &str, origin: &str) -> ConsentRecord {
        ConsentRecord {
            provider_id: id.into(),
            origin: origin.into(),
            consented_by: "me".into(),
            consented_at: "t".into(),
            credential_ref: Some("none".into()),
            ..Default::default()
        }
    }

    #[test]
    fn consent_counts_only_for_the_current_origin_of_its_scope() {
        let inst = vec![instance("ds", "https://a.example.com")];
        let ok = StoreCatalog::new(inst.clone(), &[consent("ds", "https://a.example.com")]);
        assert!(ok.is_allowed_for_project("ds"));
        let moved = StoreCatalog::new(inst.clone(), &[consent("ds", "https://old.example.com")]);
        assert!(!moved.is_allowed_for_project("ds"), "origin changed");
        // A scope with no consent (a project that never consented, or no project and
        // nothing allowed for the sessions without one): claude-code only.
        let none = StoreCatalog::new(inst, &[]);
        assert!(!none.is_allowed_for_project("ds"));
        assert!(none.is_allowed_for_project(CLAUDE_CODE));
        assert!(none.exists(CLAUDE_CODE) && !none.exists("ghost"));
        assert_eq!(none.instance_ids(), vec!["ds".to_string()]);
    }

    #[test]
    fn roles_feed_the_resolver_project_before_global() {
        let aliases = vec![ModelAlias {
            alias: "fast".into(),
            provider: "ds".into(),
            model: "ds-fast".into(),
        }];
        let global = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: CLAUDE_CODE.into(),
                model: None,
                alias: None,
            }),
            executor: None,
        };
        let project = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: "ds".into(),
                model: None,
                alias: Some("fast".into()),
            }),
            executor: None,
        };
        let catalog = StoreCatalog::new(
            vec![instance("ds", "https://a.example.com")],
            &[consent("ds", "https://a.example.com")],
        );
        let input = resolve_input(Role::Pilot, None, None, None, &global, &project, &aliases);
        let choice = resolve(&input, &catalog).unwrap();
        assert_eq!(
            (choice.provider_id.as_str(), choice.model.as_deref()),
            ("ds", Some("ds-fast"))
        );
        assert_eq!(choice.routed_by, RoutedBy::ProjectRule);
        // Without a project role the global one applies; none at all: claude-code.
        let input = resolve_input(
            Role::Pilot,
            None,
            None,
            None,
            &global,
            &RoleAssignments::default(),
            &aliases,
        );
        assert_eq!(
            resolve(&input, &catalog).unwrap().routed_by,
            RoutedBy::GlobalRule
        );
        let input = resolve_input(
            Role::Pilot,
            None,
            None,
            None,
            &RoleAssignments::default(),
            &RoleAssignments::default(),
            &aliases,
        );
        assert_eq!(
            resolve(&input, &catalog).unwrap().routed_by,
            RoutedBy::BuiltinClaudeCode
        );
    }

    #[test]
    fn a_rule_naming_an_unconsented_instance_is_refused_for_a_pilot() {
        let project = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: "ds".into(),
                model: None,
                alias: None,
            }),
            executor: None,
        };
        let catalog = StoreCatalog::new(vec![instance("ds", "https://a.example.com")], &[]);
        let input = resolve_input(
            Role::Pilot,
            None,
            None,
            None,
            &RoleAssignments::default(),
            &project,
            &[],
        );
        assert_eq!(
            resolve(&input, &catalog).unwrap_err().code(),
            "endpoint_not_allowed"
        );
    }

    #[test]
    fn without_the_claude_cli_claude_code_is_no_default_but_stays_nameable() {
        let none = RoleAssignments::default();
        let catalog = StoreCatalog::new(
            vec![instance("local", "http://127.0.0.1:9")],
            &[consent("local", "http://127.0.0.1:9")],
        )
        .with_claude_cli(false);
        // Nothing named: the only allowed instance, not a missing CLI (B2).
        let input = resolve_input(Role::Pilot, None, None, None, &none, &none, &[]);
        let choice = resolve(&input, &catalog).unwrap();
        assert_eq!(
            (choice.provider_id.as_str(), choice.routed_by),
            ("local", RoutedBy::SoleInstance)
        );
        // Named: claude-code is still chosen, the opening tells the CLI is missing (424).
        let input = resolve_input(
            Role::Pilot,
            Some(CLAUDE_CODE),
            None,
            None,
            &none,
            &none,
            &[],
        );
        assert_eq!(resolve(&input, &catalog).unwrap().provider_id, CLAUDE_CODE);
        // With the CLI, nothing changes: claude-code is the default.
        let catalog = StoreCatalog::new(
            vec![instance("local", "http://127.0.0.1:9")],
            &[consent("local", "http://127.0.0.1:9")],
        )
        .with_claude_cli(true);
        let input = resolve_input(Role::Pilot, None, None, None, &none, &none, &[]);
        assert_eq!(
            resolve(&input, &catalog).unwrap().routed_by,
            RoutedBy::BuiltinClaudeCode
        );
    }

    #[test]
    fn an_undefined_alias_yields_no_candidate() {
        let t = RoleTarget {
            provider: "ds".into(),
            model: None,
            alias: Some("ghost".into()),
        };
        assert!(candidate_of(&t, &[]).is_none());
    }

    #[test]
    fn the_task_alias_beats_the_run_which_beats_the_project_rule() {
        let aliases = vec![ModelAlias {
            alias: "deep".into(),
            provider: "ds".into(),
            model: "ds-deep".into(),
        }];
        let project = RoleAssignments {
            pilot: Some(RoleTarget {
                provider: CLAUDE_CODE.into(),
                model: None,
                alias: None,
            }),
            executor: Some(RoleTarget {
                provider: CLAUDE_CODE.into(),
                model: None,
                alias: None,
            }),
        };
        let catalog = StoreCatalog::new(
            vec![
                instance("ds", "https://a.example.com"),
                instance("other", "https://b.example.com"),
            ],
            &[
                consent("ds", "https://a.example.com"),
                consent("other", "https://b.example.com"),
            ],
        );
        let none = RoleAssignments::default();
        let by_run = resolve_input(
            Role::Executor,
            None,
            None,
            Some(("other", Some("o-1"))),
            &none,
            &project,
            &aliases,
        );
        let c = resolve(&by_run, &catalog).unwrap();
        assert_eq!(
            (c.provider_id.as_str(), c.routed_by),
            ("other", RoutedBy::Run)
        );
        let by_task = resolve_input(
            Role::Executor,
            None,
            Some("deep"),
            Some(("other", None)),
            &none,
            &project,
            &aliases,
        );
        let c = resolve(&by_task, &catalog).unwrap();
        assert_eq!(
            (c.provider_id.as_str(), c.model.as_deref(), c.routed_by),
            ("ds", Some("ds-deep"), RoutedBy::Task)
        );
        // An explicit request still wins over both.
        let by_request = resolve_input(
            Role::Executor,
            Some("other"),
            Some("deep"),
            None,
            &none,
            &project,
            &aliases,
        );
        assert_eq!(
            resolve(&by_request, &catalog).unwrap().routed_by,
            RoutedBy::Request
        );
    }

    #[test]
    fn the_policy_is_off_shadow_or_enforce() {
        let aliases = vec![ModelAlias {
            alias: "fast".into(),
            provider: "ds".into(),
            model: "ds-fast".into(),
        }];
        let mut policy = ModelPolicy::default();
        policy.rules.insert("chat".into(), "fast".into());
        policy.rules.insert("runner.simple".into(), "fast".into());
        assert_eq!(
            policy_pick(&policy, Role::Pilot, None, &aliases),
            None,
            "off says nothing"
        );
        policy.mode = "shadow".into();
        let shadow = policy_pick(&policy, Role::Pilot, None, &aliases).unwrap();
        assert!(!shadow.enforced);
        assert_eq!(shadow.rule, "chat");
        policy.mode = "enforce".into();
        let run = policy_pick(&policy, Role::Executor, Some("simple"), &aliases).unwrap();
        assert!(run.enforced);
        assert_eq!(
            (run.rule.as_str(), run.candidate.provider_id.as_str()),
            ("runner.simple", "ds")
        );
        // An executor of an unknown class, or a rule without an alias: nothing.
        assert_eq!(
            policy_pick(&policy, Role::Executor, Some("weird"), &aliases),
            None
        );
        assert_eq!(
            policy_pick(&policy, Role::Executor, Some("complex"), &aliases),
            None
        );
        // A rule naming an undefined alias matches nothing.
        assert_eq!(policy_pick(&policy, Role::Pilot, None, &[]), None);
    }

    #[test]
    fn auto_consent_bypasses_per_project_consent() {
        let mut nim = instance("nim", "https://nim.example.com");
        nim.auto_consent = true;
        let regular = instance("ds", "https://a.example.com");
        let catalog = StoreCatalog::new(vec![nim, regular], &[]);
        // An auto-consent instance is allowed without a consent record.
        assert!(catalog.is_allowed_for_project("nim"));
        // A normal instance without consent is NOT allowed.
        assert!(!catalog.is_allowed_for_project("ds"));
        // claude-code is always allowed regardless.
        assert!(catalog.is_allowed_for_project(CLAUDE_CODE));
    }

    #[test]
    fn auto_consent_false_still_needs_consent() {
        let regular = instance("ds", "https://a.example.com");
        assert!(!regular.auto_consent); // default is false
        let without = StoreCatalog::new(vec![regular.clone()], &[]);
        assert!(!without.is_allowed_for_project("ds"));
        let with = StoreCatalog::new(vec![regular], &[consent("ds", "https://a.example.com")]);
        assert!(with.is_allowed_for_project("ds"));
    }
}
