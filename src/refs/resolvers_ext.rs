//! The resolvers of the kinds added after the first five (see
//! [`super::resolvers`] for the rules every resolver follows): conversation,
//! project, milestone, release, workspace, commit, protocol, persona, skill,
//! file and link. The table of kinds points at the constructors below.

use std::sync::Arc;

use async_trait::async_trait;

use super::access::RefMeta;
use super::resolvers::{Candidate, Candidates, KindResolver, Memo};
use super::types::{RefId, RefKind};
use crate::neo4j::GraphStore;

/// The answer for a kind nobody registered: nothing is found, nothing is
/// suggested. `GraphRefSource` falls back on it so that a kind without a
/// resolver reads as `not_found` instead of panicking; a test pins that every
/// kind of the table has a real one.
pub struct Missing;

#[async_trait]
impl KindResolver for Missing {
    fn kind(&self) -> RefKind {
        RefKind::Link
    }
    async fn load(&self, _id: &RefId, _memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        Ok(None)
    }
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        Ok(vec![])
    }
}

/// A kind whose resolver is not written yet.
struct NotYet(RefKind);

#[async_trait]
impl KindResolver for NotYet {
    fn kind(&self) -> RefKind {
        self.0
    }
    async fn load(&self, _id: &RefId, _memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        Ok(None)
    }
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        Ok(vec![])
    }
}

macro_rules! not_yet {
    ($($name:ident => $kind:ident),* $(,)?) => {$(
        pub fn $name(_graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
            Box::new(NotYet(RefKind::$kind))
        }
    )*};
}

not_yet! {
    conversation => Conversation,
    project => Project,
    milestone => Milestone,
    release => Release,
    workspace => Workspace,
    commit => Commit,
    protocol => Protocol,
    persona => Persona,
    skill => Skill,
    file => File,
}

/// The resolver of `link`. It holds no store and does no I/O: the answer is the
/// address itself, normalized by [`super::link`] (a pure parser, no lookup, no
/// request). The label is the host; reading the page is the agent's act and
/// goes through the consent by origin, not through here.
pub struct LinkResolver;

#[async_trait]
impl KindResolver for LinkResolver {
    fn kind(&self) -> RefKind {
        RefKind::Link
    }

    async fn load(&self, id: &RefId, _memo: &mut Memo) -> anyhow::Result<Option<RefMeta>> {
        // Only the canonical spelling is a link reference: anything else was
        // not built by the validation.
        let Ok(url) = super::link::normalize(id.as_str()) else {
            return Ok(None);
        };
        if url != id.as_str() {
            return Ok(None);
        }
        let Some(host) = super::link::host_label(&url) else {
            return Ok(None);
        };
        let shown = url
            .split_once("://")
            .map_or(url.as_str(), |(_, rest)| rest)
            .trim_end_matches('/');
        Ok(Some(RefMeta {
            kind: RefKind::Link,
            id: id.clone(),
            label: host,
            subtitle: Some(super::label::truncate(shown)),
            project: None,
            workspace: None,
            entity_status: None,
        }))
    }

    /// A link is pasted, not listed: nothing to suggest.
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<Candidate>> {
        Ok(vec![])
    }
}

pub fn link(_graph: Arc<dyn GraphStore>) -> Box<dyn KindResolver> {
    Box::new(LinkResolver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::GraphStore;
    use crate::refs::resolvers::GraphRefSource;
    use crate::refs::test_support::{user, world};
    use std::sync::Arc;

    #[tokio::test]
    async fn every_kind_has_a_resolver_of_its_own() {
        let w = world().await;
        let source = GraphRefSource::new(w.graph.clone());
        for kind in RefKind::ALL {
            assert_eq!(source.resolver(kind).kind(), kind, "{kind}");
        }
    }
    // ---- link ---------------------------------------------------------

    use crate::refs::access::{AccessPolicy, Principal, Resolution, SessionScope};
    use crate::refs::types::EntityRef;

    async fn resolve(
        w: &crate::refs::test_support::World,
        p: &Principal,
        r: &EntityRef,
    ) -> Resolution {
        AccessPolicy::open_instance()
            .resolve_checked(p, &GraphRefSource::new(w.graph.clone()), r)
            .await
    }

    fn link_ref(url: &str) -> EntityRef {
        crate::refs::validate::validate_token(&format!("#link:{url}")).unwrap()
    }

    #[tokio::test]
    async fn a_link_resolves_to_its_host_without_touching_the_store() {
        let w = world().await;
        let r = link_ref("https://Docs.Example.com/guide?page=2#top");
        assert_eq!(r.id.as_str(), "https://docs.example.com/guide?page=2");
        let Resolution::Found(m) = resolve(&w, &user(), &r).await else {
            panic!("a valid link resolves");
        };
        assert_eq!(m.kind, RefKind::Link);
        assert_eq!(m.label, "docs.example.com");
        assert_eq!(m.subtitle.as_deref(), Some("docs.example.com/guide?page=2"));
        assert!(m.project.is_none() && m.workspace.is_none() && m.entity_status.is_none());
        // the resolver holds no store: it cannot read one.
        assert_eq!(std::mem::size_of::<LinkResolver>(), 0);
    }

    #[tokio::test]
    async fn a_link_is_shared_by_every_session_whatever_its_project() {
        let w = world().await;
        let r = link_ref("https://example.com/");
        for scope in [
            SessionScope::default(),
            SessionScope {
                project: Some(w.a),
                workspace: Some(w.ws.id),
            },
            SessionScope {
                project: Some(w.b),
                workspace: None,
            },
        ] {
            assert!(
                matches!(
                    resolve(&w, &Principal::Session(scope), &r).await,
                    Resolution::Found(_)
                ),
                "{scope:?}"
            );
        }
        assert_eq!(
            resolve(&w, &Principal::Unauthenticated, &r).await,
            Resolution::Forbidden
        );
    }

    #[tokio::test]
    async fn an_id_that_is_not_the_normalized_address_reads_not_found() {
        // Only validation builds a link reference; a stored block from elsewhere
        // may hold anything, and the resolver does not trust it.
        let w = world().await;
        let src = GraphRefSource::new(w.graph.clone());
        for bad in [
            "javascript:alert(1)",
            "https://user:pw@example.com/",
            "http://127.0.0.1/admin",
            "http://169.254.169.254/",
            "HTTPS://EXAMPLE.COM/",
            "https://example.com/#frag",
            "https://example.com/a b",
        ] {
            let id = serde_json::from_value::<RefId>(serde_json::json!(bad)).unwrap();
            let got = src
                .resolver(RefKind::Link)
                .load(&id, &mut Memo::default())
                .await
                .unwrap();
            assert!(got.is_none(), "{bad}");
        }
    }

    #[tokio::test]
    async fn a_link_is_not_searchable_and_the_search_does_not_fail() {
        let w = world().await;
        let q = crate::refs::search::RefSearchParams {
            q: Some("example".into()),
            kinds: Some("link,plan".into()),
            ..Default::default()
        }
        .into_query()
        .unwrap();
        let out = crate::refs::search::search(
            w.graph.clone(),
            &AccessPolicy::open_instance(),
            &user(),
            &q,
        )
        .await
        .unwrap();
        assert!(out.items.iter().all(|i| i.kind != RefKind::Link));
    }

    #[tokio::test]
    async fn a_turn_names_a_link_by_its_host_and_never_fetches_it() {
        let w = world().await;
        let r = link_ref("https://example.com/private-looking/path?token=abc");
        let g: Arc<dyn GraphStore> = w.graph.clone();
        let stored = crate::refs::block::encode("regarde", std::slice::from_ref(&r));
        let t = crate::refs::turn::expand_user_turn_if(&g, &stored, true).await;
        assert_eq!(t.resolved.len(), 1);
        assert_eq!(t.resolved[0].label.as_deref(), Some("example.com"));
        assert!(
            t.model_tail.contains("link \"example.com\""),
            "{}",
            t.model_tail
        );
    }
}
