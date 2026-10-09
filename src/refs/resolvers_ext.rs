//! The resolvers of the kinds added after the first five (see
//! [`super::resolvers`] for the rules every resolver follows): conversation,
//! project, milestone, release, workspace, commit, protocol, persona, skill,
//! file and link. The table of kinds points at the constructors below.

use std::sync::Arc;

use async_trait::async_trait;

use super::access::RefMeta;
use super::resolvers::{Candidates, KindResolver, Memo};
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
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
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
    async fn candidates(&self, _c: &Candidates, _m: &mut Memo) -> anyhow::Result<Vec<RefMeta>> {
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
    link => Link,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refs::resolvers::GraphRefSource;
    use crate::refs::test_support::world;

    #[tokio::test]
    async fn every_kind_has_a_resolver_of_its_own() {
        let w = world().await;
        let source = GraphRefSource::new(w.graph.clone());
        for kind in RefKind::ALL {
            assert_eq!(source.resolver(kind).kind(), kind, "{kind}");
        }
    }
}
