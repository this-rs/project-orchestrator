//! Knowledge Notes module
//!
//! Provides a system for capturing and managing contextual knowledge from conversations:
//! guidelines, gotchas, patterns, tips, and verifiable assertions.
//!
//! Notes can be linked to code entities and automatically surfaced to agents
//! based on relevance and graph propagation.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod beliefs;
pub mod coactivation;
pub mod hashing;
pub mod lifecycle;
pub mod manager;
pub mod models;
pub mod retrieval_ledger;
pub mod retrieval_policy;
pub mod witness;

/// Time constant tau (days) of the stored energy decay `exp(-t/tau)` (halves every
/// tau·ln2 ≈ 62.4 days; NOT a half-life, despite the name) — the single value used by the
/// heartbeat, skill maintenance and data migrations (it used to be 14 days
/// in one place and 90 in others).
pub const ENERGY_HALF_LIFE_DAYS: f64 = 90.0;

/// Days without human activity (code sync, chat) after which a project is
/// dormant: its notes' energy and synapses stop decaying until it is active
/// again, so knowledge survives months without work on the project.
pub const PROJECT_DORMANT_AFTER_DAYS: i64 = 14;

/// Whether a note is knowledge the agent may still be given: active or
/// awaiting review, and not replaced by a newer note. Archived, obsolete
/// (invalidated), stale and superseded notes must never reach its context —
/// newer knowledge wins.
///
/// Audit-trail notes written by skill evolution (`created_by =
/// "skill-evolution"`, e.g. "Skill X orphaned and archived") are a log, not
/// knowledge: never given to the agent either.
pub fn is_current_knowledge(note: &Note) -> bool {
    is_current_knowledge_at(note, chrono::Utc::now())
}

/// [`is_current_knowledge`] at a given instant — the form tests and replays use.
///
/// A note outside its validity window is NOT current: it is given to the agent only
/// after a revision (a new version carrying its provenance), never as it stands.
pub fn is_current_knowledge_at(note: &Note, now: chrono::DateTime<chrono::Utc>) -> bool {
    matches!(note.status, NoteStatus::Active | NoteStatus::NeedsReview)
        && note.superseded_by.is_none()
        && note.created_by != SKILL_EVOLUTION_AUTHOR
        && note.validity_at(now) == Validity::Current
}

/// `created_by` of the audit-trail notes skill evolution writes.
pub const SKILL_EVOLUTION_AUTHOR: &str = "skill-evolution";

pub use hashing::*;
pub use lifecycle::*;
pub use manager::{BackfillProgress, NoteManager, SynapseBackfillProgress, SynapseConfig};
pub use models::*;
pub use witness::{CodeRef, ExternalRef, Witness, WitnessKind, WitnessValidationError};

#[cfg(test)]
mod current_knowledge_tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    #[test]
    fn test_only_current_real_knowledge_reaches_the_agent() {
        let mk = || Note::new(None, NoteType::Gotcha, "x".into(), "user".into());
        assert!(is_current_knowledge(&mk()));
        let mut archived = mk();
        archived.status = NoteStatus::Archived;
        assert!(!is_current_knowledge(&archived));
        let mut replaced = mk();
        replaced.superseded_by = Some(uuid::Uuid::new_v4());
        assert!(!is_current_knowledge(&replaced));
        let trace = Note::new(
            None,
            NoteType::Observation,
            "Skill X orphaned".into(),
            SKILL_EVOLUTION_AUTHOR.into(),
        );
        assert!(
            !is_current_knowledge(&trace),
            "evolution audit trail is not knowledge"
        );
    }
}

pub mod provenance {
    //! Stable agent provenance: who, among the agents, wrote something to the graph.
    //!
    //! A write made by a chat session used to read `created_by: "api"`, like a script's.
    //! The identity of the writer is the DURABLE SELF: the id is the root UUID the graph
    //! creates ONCE and keeps ([`crate::neo4j::AnalyticsStore::agent_identity_namespace`]),
    //! stable across sessions, models and restarts, private to this store.
    //!
    //! The model is NOT the identity. It is a dated fact beside it: a `(:Agent {kind:'model'})`
    //! facet, attached to the self by `FACET_OF`, carries the model name and `last_seen`.
    //! Changing model changes the facet read, never the id that writes.
    //!
    //! Written as `agent:<uuid>` where an actor string already lives (`created_by`), so
    //! the values written before (`api`, `system-inference`, `skill-evolution`, ...) stay
    //! as they are and read as before: [`parse_actor`] says `None` for them. Actors written
    //! before this change named the MODEL facet; the migration of
    //! [`crate::neo4j::ChatStore`] moves their edges to the self.
    //!
    //! Limits: the identity is the store's self, not the person or persona behind it; ids
    //! mean nothing on another instance; with authentication disabled the calling session
    //! is a header, a hint and not a proof (`envelope::identify_caller`).

    use anyhow::Result;
    use uuid::Uuid;

    use crate::neo4j::GraphStore;

    /// Prefix of an agent actor in a `created_by`.
    pub const AGENT_ACTOR_PREFIX: &str = "agent:";

    /// Model names that do not name a model: a routing mode or a tier the provider
    /// resolves (`auto`, `default`, ...) and the family aliases of Claude Code. A write
    /// stamped with one of them would merge unrelated models into one "agent".
    const NOT_A_MODEL: &[&str] = &[
        "auto", "default", "fast", "deep", "utility", "best", "opus", "sonnet", "haiku", "opusplan",
    ];

    /// The model name as a fact: trimmed, lower-cased, without the context-window suffix
    /// (`[1m]`, as `agent_hooks::same_model` reads it). `None` for an empty model or a name
    /// that is not a model ([`NOT_A_MODEL`]): it names no facet and no writer.
    pub fn normalized_model(model: &str) -> Option<String> {
        let model = model.trim().to_ascii_lowercase();
        let model = model
            .strip_suffix("[1m]")
            .unwrap_or(&model)
            .trim_end()
            .to_string();
        if model.is_empty() || NOT_A_MODEL.contains(&model.as_str()) {
            return None;
        }
        Some(model)
    }

    /// The durable self that writes for a session running `model`, under the store's
    /// `namespace`. The model only decides WHETHER a writer can be named (a session that
    /// names no model names none); the id returned is the self, whatever the model.
    /// `None` keeps the caller's fallback — never another agent's id.
    pub fn agent_id(namespace: Uuid, model: &str) -> Option<Uuid> {
        normalized_model(model).map(|_| namespace)
    }

    /// The id of the model facet of `model` under the store's `namespace`: the dated fact
    /// that this self ran on that model. A facet is never the writer (see [`agent_id`]).
    pub fn model_facet_id(namespace: Uuid, model: &str) -> Option<Uuid> {
        normalized_model(model).map(|m| Uuid::new_v5(&namespace, format!("facet:{m}").as_bytes()))
    }

    /// The `created_by` of a write made by the agent `id`.
    pub fn actor(id: Uuid) -> String {
        format!("{AGENT_ACTOR_PREFIX}{id}")
    }

    /// The agent behind a `created_by`, if an agent wrote it. Every other value (`api`,
    /// `system/...`, a person) is `None`: it was not an agent, or not one we can name.
    pub fn parse_actor(created_by: &str) -> Option<Uuid> {
        created_by
            .strip_prefix(AGENT_ACTOR_PREFIX)
            .and_then(|id| Uuid::parse_str(id).ok())
    }

    /// The durable self of the chat session `session_id`, with the model it runs on (a
    /// dated fact, not the identity). `Ok(None)` for a session that does not exist or
    /// names no model: no agent can be named for it.
    pub async fn self_of_session(
        graph: &dyn GraphStore,
        session_id: Uuid,
    ) -> Result<Option<(Uuid, String)>> {
        let Some(session) = graph.get_chat_session(session_id).await? else {
            return Ok(None);
        };
        let namespace = graph.agent_identity_namespace().await?;
        Ok(agent_id(namespace, &session.model).map(|id| (id, session.model.clone())))
    }

    /// The durable self of the chat session `session_id`. See [`self_of_session`].
    pub async fn agent_id_of_session(
        graph: &dyn GraphStore,
        session_id: Uuid,
    ) -> Result<Option<Uuid>> {
        Ok(self_of_session(graph, session_id).await?.map(|(id, _)| id))
    }

    /// The `created_by` a write must carry: the agent actor of the calling session when
    /// one can be named, `fallback` otherwise. Never fails: provenance is not a reason to
    /// refuse a write — a session or root that cannot be read is logged and the write
    /// keeps the fallback it always had.
    pub async fn author_of(
        graph: &dyn GraphStore,
        calling_session: Option<&str>,
        fallback: &str,
    ) -> String {
        let Some(session) = calling_session.and_then(|s| Uuid::parse_str(s.trim()).ok()) else {
            return fallback.to_string();
        };
        match self_of_session(graph, session).await {
            Ok(Some((id, model))) => {
                // The self becomes a node the first time it writes, and the model of the
                // session is recorded beside it as a dated fact. Never a reason to refuse
                // the write: a graph that cannot make the node is logged.
                if let Err(e) = graph.ensure_agent(id, Some(&model)).await {
                    tracing::warn!(agent = %id, error = %e, "the node of an agent could not be made");
                }
                actor(id)
            }
            Ok(None) => fallback.to_string(),
            Err(e) => {
                tracing::warn!(session_id = %session, error = %e, "the agent behind a write could not be named");
                fallback.to_string()
            }
        }
    }

    /// Records, durably, that the agent of `session` applied `label` (a persona, a skill)
    /// — the edge the activation registry only ever held in memory.
    ///
    /// Best effort by construction: an agent that cannot be named, or a target the graph
    /// does not have, draws nothing and is logged. Activation decides what the turn sees;
    /// it must never fail because a bookkeeping edge could not be written.
    pub async fn record_agent_uses(
        graph: &dyn GraphStore,
        session_id: Uuid,
        label: &str,
        target: Uuid,
    ) {
        let agent = match agent_id_of_session(graph, session_id).await {
            Ok(Some(id)) => id,
            Ok(None) => return,
            Err(e) => {
                tracing::debug!(session_id = %session_id, error = %e, "the agent of an activation could not be named");
                return;
            }
        };
        if let Err(e) = graph.ensure_agent(agent, None).await {
            tracing::debug!(agent = %agent, error = %e, "the node of an activated agent could not be made");
        }
        // La preuve de provenance est faite ICI, avant la requête : une règle nommée et une
        // mesure datée, sinon l'écrivain n'est même pas appelable.
        let provenance = match crate::graph::rules::AccountableProvenance::new(
            crate::graph::rules::AGENT_USES_ACTIVATED.name,
            format!("activé dans la session {session_id}"),
        ) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(agent = %agent, label, error = %e, "an activation has no accountable provenance");
                return;
            }
        };
        if let Err(e) = graph
            .link_agent_uses(agent, label, target, &provenance)
            .await
        {
            tracing::debug!(agent = %agent, label, error = %e, "an activation did not become an edge");
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::neo4j::mock::MockGraphStore;

        /// An activation is written on the agent of its session, with its provenance.
        #[tokio::test]
        async fn an_activation_becomes_an_edge_of_the_agent_that_applied_it() {
            use crate::neo4j::traits::*;

            let graph = MockGraphStore::new();
            let session = crate::test_helpers::test_chat_session(None);
            let sid = session.id;
            graph.create_chat_session(&session).await.unwrap();
            let persona = Uuid::new_v4();

            record_agent_uses(&graph, sid, "Persona", persona).await;

            let uses = graph.agent_uses.read().await;
            let key = uses
                .keys()
                .find(|(_, label, target)| label == "Persona" && *target == persona)
                .expect("the activation is an edge");
            let provenance = uses.get(key).unwrap();
            assert!(provenance.is_accountable());
            assert_eq!(
                provenance.rule,
                crate::graph::rules::AGENT_USES_ACTIVATED.name
            );
        }

        /// A session the graph does not know writes nothing — « never fail without a socle »:
        /// a bookkeeping edge must not be able to fail a turn.
        #[tokio::test]
        async fn an_activation_of_an_unknown_session_writes_nothing() {
            let graph = MockGraphStore::new();
            record_agent_uses(&graph, Uuid::new_v4(), "Persona", Uuid::new_v4()).await;
            assert!(graph.agent_uses.read().await.is_empty());
        }
        #[allow(unused_imports)]
        use crate::neo4j::traits::*;

        async fn session_of(graph: &MockGraphStore, model: &str) -> Uuid {
            let mut s = crate::test_helpers::test_chat_session(Some("alpha"));
            s.model = model.into();
            let id = s.id;
            graph.create_chat_session(&s).await.unwrap();
            id
        }

        /// One self writes for the whole store, whatever the model of the session: the node
        /// of the writer is made once, and a session naming no model keeps its fallback.
        #[tokio::test]
        async fn a_write_makes_the_durable_self_a_node_once() {
            let graph = MockGraphStore::new();
            let a = session_of(&graph, "claude-opus-5-5").await;
            let a_again = session_of(&graph, " Claude-Opus-5-5 ").await;
            let other = session_of(&graph, "deepseek-v4").await;
            let none = session_of(&graph, "auto").await;

            let by_a = author_of(&graph, Some(&a.to_string()), "api").await;
            assert_eq!(
                author_of(&graph, Some(&a_again.to_string()), "api").await,
                by_a
            );
            assert_eq!(graph.agents.lock().unwrap().len(), 1, "one self, one node");
            assert_eq!(
                author_of(&graph, Some(&other.to_string()), "api").await,
                by_a,
                "another model writes as the same self"
            );
            assert_eq!(graph.agents.lock().unwrap().len(), 1);

            // A session that names no model leaves no node and keeps its fallback.
            assert_eq!(
                author_of(&graph, Some(&none.to_string()), "api").await,
                "api"
            );
            assert_eq!(graph.agents.lock().unwrap().len(), 1);
            assert_eq!(
                parse_actor(&by_a),
                Some(graph.agent_identity_namespace().await.unwrap())
            );
        }

        /// Changing the model of a session changes the dated facet, never the writer.
        #[tokio::test]
        async fn changing_model_does_not_change_the_identity_that_writes() {
            let graph = MockGraphStore::new();
            let sid = session_of(&graph, "claude-opus-5-5").await;
            let before = author_of(&graph, Some(&sid.to_string()), "api").await;

            graph
                .update_chat_session_model(sid, "deepseek-v4")
                .await
                .unwrap();
            let after = author_of(&graph, Some(&sid.to_string()), "api").await;
            assert_eq!(before, after, "the writer is the self, not the model");
            assert_eq!(
                agent_id_of_session(&graph, sid).await.unwrap(),
                parse_actor(&after),
            );

            // The model stays a dated fact: a facet per name, the same for a re-spelling.
            let ns = graph.agent_identity_namespace().await.unwrap();
            assert_ne!(
                model_facet_id(ns, "claude-opus-5-5"),
                model_facet_id(ns, "deepseek-v4")
            );
            assert_eq!(
                model_facet_id(ns, " Claude-Opus-5-5[1m] "),
                model_facet_id(ns, "claude-opus-5-5")
            );
        }

        /// Two agents of one store do not read each other's knowledge: a read names one
        /// agent, and only that agent's edges answer it.
        #[tokio::test]
        async fn two_distinct_agents_do_not_see_each_others_knowledge() {
            use crate::graph::rules::{AccountableProvenance, KnowledgeScope};
            let graph = MockGraphStore::new();
            let note = crate::test_helpers::test_note(
                Uuid::new_v4(),
                crate::notes::NoteType::Gotcha,
                "un savoir",
            );
            graph.create_note(&note).await.unwrap();
            let me = graph.agent_identity_namespace().await.unwrap();
            let other = Uuid::new_v4();
            let provenance =
                AccountableProvenance::new("agent-knows-core", "1 portée, 1 note").unwrap();

            graph
                .link_agent_knows(other, note.id, KnowledgeScope::Global, None, &provenance)
                .await
                .unwrap();
            assert!(graph
                .agent_knows(me, None, 10)
                .await
                .unwrap()
                .notes
                .is_empty());
            assert_eq!(
                graph.agent_knows(other, None, 10).await.unwrap().notes,
                vec![note.id]
            );

            // Another store is another root, so another agent altogether.
            let elsewhere = MockGraphStore::new();
            assert_ne!(elsewhere.agent_identity_namespace().await.unwrap(), me);
        }

        /// An agent that cannot be resolved receives the fallback and no other agent's
        /// context: an unknown session, one naming no model, and an unknown agent read.
        #[tokio::test]
        async fn an_unresolved_agent_gets_no_other_agents_context() {
            let graph = MockGraphStore::new();
            let unknown = Uuid::new_v4();
            assert_eq!(
                author_of(&graph, Some(&unknown.to_string()), "api").await,
                "api"
            );
            let no_model = session_of(&graph, "auto").await;
            assert_eq!(
                author_of(&graph, Some(&no_model.to_string()), "api").await,
                "api"
            );
            assert!(graph.agents.lock().unwrap().is_empty());
            let read = graph.agent_knows(unknown, None, 10).await.unwrap();
            assert!(read.notes.is_empty() && read.available == 0);
        }

        #[tokio::test]
        async fn the_root_is_created_once_and_kept() {
            let graph = MockGraphStore::new();
            let first = graph.agent_identity_namespace().await.unwrap();
            let second = graph.agent_identity_namespace().await.unwrap();
            assert_eq!(first, second);
            // Another store, another root: its agents are other agents.
            let elsewhere = MockGraphStore::new()
                .agent_identity_namespace()
                .await
                .unwrap();
            assert_ne!(first, elsewhere);
        }

        #[tokio::test]
        async fn a_write_without_a_nameable_agent_keeps_its_old_author() {
            let graph = MockGraphStore::new();
            assert_eq!(author_of(&graph, None, "api").await, "api");
            assert_eq!(author_of(&graph, Some("not-a-uuid"), "api").await, "api");
            let unknown = Uuid::new_v4().to_string();
            assert_eq!(author_of(&graph, Some(&unknown), "api").await, "api");
            let no_model = session_of(&graph, "  ").await.to_string();
            assert_eq!(author_of(&graph, Some(&no_model), "api").await, "api");

            let sid = session_of(&graph, "claude-opus-5-5").await;
            let author = author_of(&graph, Some(&sid.to_string()), "api").await;
            assert_eq!(
                parse_actor(&author),
                agent_id_of_session(&graph, sid).await.unwrap()
            );
        }

        #[test]
        fn a_routing_mode_or_an_alias_names_no_agent_and_the_window_suffix_is_not_a_model() {
            let ns = Uuid::new_v4();
            for not_a_model in ["auto", "Default", " opus ", "sonnet", "haiku", "fast", ""] {
                assert_eq!(agent_id(ns, not_a_model), None, "{not_a_model:?}");
            }
            assert_eq!(
                agent_id(ns, "claude-opus-4-6[1m]"),
                agent_id(ns, "claude-opus-4-6")
            );
            assert_eq!(
                agent_id(ns, "Claude-Opus-4-6[1M]"),
                agent_id(ns, "claude-opus-4-6")
            );
            assert!(agent_id(ns, "claude-opus-4-6").is_some());
        }

        #[test]
        fn old_authors_still_read_and_are_not_agents() {
            for old in [
                "api",
                "system-inference",
                "skill-evolution",
                "system/heartbeat",
                "agent:",
                "agent:nope",
            ] {
                assert_eq!(parse_actor(old), None, "{old}");
            }
            let id = Uuid::new_v4();
            assert_eq!(parse_actor(&actor(id)), Some(id));
        }
    }
}

#[cfg(test)]
mod validity_window_tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn mk() -> Note {
        Note::new(None, NoteType::Gotcha, "x".into(), "user".into())
    }

    #[test]
    fn a_note_without_a_window_is_unchanged() {
        // The whole point of PR5 being safe: no window, no new behaviour.
        let n = mk();
        assert_eq!(n.validity_at(Utc::now()), Validity::Current);
        assert!(is_current_knowledge(&n));
    }

    #[test]
    fn a_note_past_its_valid_until_is_not_current() {
        let now = Utc::now();
        let mut n = mk();
        n.valid_until = Some(now - Duration::days(1));
        assert_eq!(
            n.validity_at(now),
            Validity::Expired(now - Duration::days(1))
        );
        assert!(!is_current_knowledge_at(&n, now));
    }

    #[test]
    fn a_note_before_its_valid_from_is_not_current() {
        let now = Utc::now();
        let mut n = mk();
        n.valid_from = Some(now + Duration::days(1));
        assert!(matches!(n.validity_at(now), Validity::NotYetValid(_)));
        assert!(!is_current_knowledge_at(&n, now));
    }

    #[test]
    fn the_window_is_closed_at_the_start_and_open_at_the_end() {
        let t0 = Utc::now();
        let t1 = t0 + Duration::days(10);
        let mut n = mk();
        n.valid_from = Some(t0);
        n.valid_until = Some(t1);
        assert!(is_current_knowledge_at(&n, t0), "valid_from is inclusive");
        assert!(is_current_knowledge_at(&n, t1 - Duration::seconds(1)));
        assert!(!is_current_knowledge_at(&n, t1), "valid_until is exclusive");
    }

    #[test]
    fn a_window_does_not_resurrect_a_superseded_note() {
        let now = Utc::now();
        let mut n = mk();
        n.valid_until = Some(now + Duration::days(30));
        n.superseded_by = Some(uuid::Uuid::new_v4());
        assert!(!is_current_knowledge_at(&n, now));
    }

    #[test]
    fn a_note_serialized_before_the_window_existed_still_reads() {
        // Stored JSON from before PR5 has none of the three fields.
        let mut v = serde_json::to_value(mk()).unwrap();
        let o = v.as_object_mut().unwrap();
        for k in ["valid_from", "valid_until", "revision"] {
            o.remove(k);
        }
        let n: Note = serde_json::from_value(v).unwrap();
        assert!(n.valid_from.is_none() && n.valid_until.is_none() && n.revision.is_none());
    }
}
