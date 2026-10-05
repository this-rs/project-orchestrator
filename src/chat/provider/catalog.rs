//! The instance catalog the resolver reads, built from what is stored
//! (instances, the consent of the project being served, roles, aliases).
//!
//! The resolver stays pure ([`super::resolver::InstanceCatalog`] is a sync
//! trait over facts); this module loads those facts once per opening, so a
//! decision is taken on one consistent view.

use std::collections::{HashMap, HashSet};

use super::resolver::{Candidate, InstanceCatalog, ResolveInput, Role, CLAUDE_CODE};
use super::settings::{ConsentRecord, InstanceRecord, ModelAlias, RoleAssignments, RoleTarget};

/// Facts about the instances for ONE project (or none).
#[derive(Debug, Clone, Default)]
pub struct StoreCatalog {
    instances: HashMap<String, InstanceRecord>,
    consented: HashSet<String>,
    unhealthy: HashSet<String>,
}

impl StoreCatalog {
    /// Builds the catalog. A consent counts only while it is tied to the
    /// instance's CURRENT origin (A28); `project_given` is whether the opening
    /// has a project at all (without one only claude-code is allowed).
    pub fn new(
        instances: Vec<InstanceRecord>,
        consents: &[ConsentRecord],
        project_given: bool,
    ) -> Self {
        let consented = if project_given {
            instances
                .iter()
                .filter(|i| {
                    consents
                        .iter()
                        .any(|c| c.provider_id == i.id && c.origin == i.origin)
                })
                .map(|i| i.id.clone())
                .collect()
        } else {
            HashSet::new()
        };
        Self {
            instances: instances.into_iter().map(|i| (i.id.clone(), i)).collect(),
            consented,
            unhealthy: HashSet::new(),
        }
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
        provider_id == CLAUDE_CODE || self.consented.contains(provider_id)
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

/// Fills the resolver input from the stored roles: the project's role is the
/// project rule, the global one the global rule (A16). An explicit request is
/// the caller's.
pub fn resolve_input<'a>(
    role: Role,
    request_provider: Option<&str>,
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
    input.project_rule = pick(project);
    input.global_rule = pick(global);
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::provider::resolver::{resolve, RoutedBy};

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
        }
    }

    fn consent(id: &str, origin: &str) -> ConsentRecord {
        ConsentRecord {
            provider_id: id.into(),
            origin: origin.into(),
            consented_by: "me".into(),
            consented_at: "t".into(),
        }
    }

    #[test]
    fn consent_counts_only_for_the_current_origin_and_with_a_project() {
        let inst = vec![instance("ds", "https://a.example.com")];
        let ok = StoreCatalog::new(
            inst.clone(),
            &[consent("ds", "https://a.example.com")],
            true,
        );
        assert!(ok.is_allowed_for_project("ds"));
        let moved = StoreCatalog::new(
            inst.clone(),
            &[consent("ds", "https://old.example.com")],
            true,
        );
        assert!(!moved.is_allowed_for_project("ds"), "origin changed");
        let no_project = StoreCatalog::new(inst, &[consent("ds", "https://a.example.com")], false);
        assert!(
            !no_project.is_allowed_for_project("ds"),
            "no project: claude-code only"
        );
        assert!(no_project.is_allowed_for_project(CLAUDE_CODE));
        assert!(no_project.exists(CLAUDE_CODE) && !no_project.exists("ghost"));
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
            true,
        );
        let input = resolve_input(Role::Pilot, None, &global, &project, &aliases);
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
        let catalog = StoreCatalog::new(vec![instance("ds", "https://a.example.com")], &[], true);
        let input = resolve_input(
            Role::Pilot,
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
    fn an_undefined_alias_yields_no_candidate() {
        let t = RoleTarget {
            provider: "ds".into(),
            model: None,
            alias: Some("ghost".into()),
        };
        assert!(candidate_of(&t, &[]).is_none());
    }
}
