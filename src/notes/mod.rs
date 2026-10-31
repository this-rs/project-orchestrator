//! Knowledge Notes module
//!
//! Provides a system for capturing and managing contextual knowledge from conversations:
//! guidelines, gotchas, patterns, tips, and verifiable assertions.
//!
//! Notes can be linked to code entities and automatically surfaced to agents
//! based on relevance and graph propagation.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

pub mod coactivation;
pub mod hashing;
pub mod lifecycle;
pub mod manager;
pub mod models;
pub mod retrieval_ledger;
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
    matches!(note.status, NoteStatus::Active | NoteStatus::NeedsReview)
        && note.superseded_by.is_none()
        && note.created_by != SKILL_EVOLUTION_AUTHOR
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
    //! A write made by a chat session used to read `created_by: "api"`, like a script's:
    //! two sessions of the same agent could not be told apart from anything else, nor
    //! recognised as the same agent. An agent's identity is its MODEL, the same in every
    //! session whatever the context: two sessions of one model are the same agent, two
    //! models are two agents. It is decoupled from projects (a model writes in many).
    //!
    //! The id is `uuid v5(root, model)`, the root a UUID the graph creates ONCE and keeps
    //! ([`crate::neo4j::AnalyticsStore::agent_identity_namespace`]): stable across sessions
    //! and restarts, private to this store (another instance's agents are other agents),
    //! and computed, never stored per write.
    //!
    //! Written as `agent:<uuid>` where an actor string already lives (`created_by`), so
    //! the values written before (`api`, `system-inference`, `skill-evolution`, ...) stay
    //! as they are and read as before: [`parse_actor`] says `None` for them.
    //!
    //! Limits, on purpose: the identity is the model, not the person or persona behind
    //! it (every Opus is one agent); a write takes the session's model AT THAT MOMENT, so
    //! a conversation that switches provider writes under two agents; ids mean nothing on
    //! another instance; with authentication disabled the calling session is a header,
    //! a hint and not a proof (`envelope::identify_caller`).

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

    /// The agent id of `model` under the store's `namespace`. Case, surrounding spaces
    /// and the context-window suffix (`[1m]`, as `agent_hooks::same_model` reads it)
    /// do not make another agent. `None` for an empty model or a name that is not a
    /// model ([`NOT_A_MODEL`]): no agent can be named, the caller keeps its fallback.
    pub fn agent_id(namespace: Uuid, model: &str) -> Option<Uuid> {
        let model = model.trim().to_ascii_lowercase();
        let model = model
            .strip_suffix("[1m]")
            .unwrap_or(&model)
            .trim_end()
            .to_string();
        if model.is_empty() || NOT_A_MODEL.contains(&model.as_str()) {
            return None;
        }
        Some(Uuid::new_v5(&namespace, model.as_bytes()))
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

    /// The agent id of the chat session `session_id`: its current model, under this
    /// store's root. `Ok(None)` for a session that does not exist or has no model.
    pub async fn agent_id_of_session(
        graph: &dyn GraphStore,
        session_id: Uuid,
    ) -> Result<Option<Uuid>> {
        let Some(session) = graph.get_chat_session(session_id).await? else {
            return Ok(None);
        };
        let namespace = graph.agent_identity_namespace().await?;
        Ok(agent_id(namespace, &session.model))
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
        match agent_id_of_session(graph, session).await {
            Ok(Some(id)) => {
                // The agent becomes a node the first time it writes. Never a reason to
                // refuse the write: a graph that cannot make the node is logged.
                if let Err(e) = graph.ensure_agent(id, None).await {
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
        let provenance = crate::graph::rules::Provenance::new(
            crate::graph::rules::AGENT_USES_ACTIVATED.name,
            format!("activé dans la session {session_id}"),
        );
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

        #[tokio::test]
        async fn a_write_makes_the_agent_a_node_once_per_identity() {
            let graph = MockGraphStore::new();
            let a = session_of(&graph, "claude-opus-5-5").await;
            let a_again = session_of(&graph, " Claude-Opus-5-5 ").await;
            let other = session_of(&graph, "deepseek-v4").await;
            let none = session_of(&graph, "auto").await;

            let by_a = author_of(&graph, Some(&a.to_string()), "api").await;
            author_of(&graph, Some(&a_again.to_string()), "api").await;
            assert_eq!(graph.agents.lock().unwrap().len(), 1, "one model, one node");
            author_of(&graph, Some(&other.to_string()), "api").await;
            assert_eq!(
                graph.agents.lock().unwrap().len(),
                2,
                "two models, two nodes"
            );

            // A session that names no model leaves no node and keeps its fallback.
            assert_eq!(
                author_of(&graph, Some(&none.to_string()), "api").await,
                "api"
            );
            assert_eq!(graph.agents.lock().unwrap().len(), 2);
            assert!(parse_actor(&by_a).is_some());
        }

        #[tokio::test]
        async fn two_sessions_of_one_model_are_the_same_agent() {
            let graph = MockGraphStore::new();
            let a = session_of(&graph, "claude-opus-5-5").await;
            let b = session_of(&graph, " Claude-Opus-5-5 ").await;
            let other = session_of(&graph, "deepseek-v4").await;
            let id_a = agent_id_of_session(&graph, a).await.unwrap().unwrap();
            let id_b = agent_id_of_session(&graph, b).await.unwrap().unwrap();
            let id_other = agent_id_of_session(&graph, other).await.unwrap().unwrap();
            assert_eq!(id_a, id_b, "the same model in two sessions is one agent");
            assert_ne!(id_a, id_other, "two models are two agents");
            // Survives the session: computed again later, from the store, it is the same.
            graph.delete_chat_session(a).await.unwrap();
            let again = agent_id(
                graph.agent_identity_namespace().await.unwrap(),
                "claude-opus-5-5",
            );
            assert_eq!(again, Some(id_a));
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
