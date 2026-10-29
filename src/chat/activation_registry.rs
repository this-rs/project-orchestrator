//! Which skills and personas were injected into each live agent.
//!
//! The enrichment stages (skill activation, persona) know the session they serve,
//! but the cockpit reads its agents through the chat manager, not through them.
//! This registry is the seam between the two: the stages record, the live snapshot
//! reads. It is in memory only and bounded to the live sessions: a session that
//! ends is forgotten the next time the snapshot is taken.
#[allow(unused_imports)]
use crate::neo4j::traits::*;

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use uuid::Uuid;

/// Most distinct references kept per kind and per session. The newest win.
const MAX_PER_KIND: usize = 8;

/// One skill or persona, as the cockpit shows it.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ActivatedRef {
    pub id: String,
    pub name: String,
}

/// What one live session has been given so far.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionActivations {
    pub skills: Vec<ActivatedRef>,
    pub personas: Vec<ActivatedRef>,
}

/// Add references to a list: an already known one moves to the front, and the
/// list stays at most `MAX_PER_KIND` long.
fn merge(list: &mut Vec<ActivatedRef>, incoming: impl IntoIterator<Item = ActivatedRef>) {
    for r in incoming {
        list.retain(|x| x.id != r.id);
        list.insert(0, r);
    }
    list.truncate(MAX_PER_KIND);
}

/// The table itself. Production uses the process-wide instance behind the free
/// functions below; tests build their own, so parallel tests never share state.
#[derive(Debug, Default)]
pub struct ActivationRegistry {
    by_session: Mutex<HashMap<Uuid, SessionActivations>>,
}

impl ActivationRegistry {
    /// Record the skills an activation stage just injected into `session`.
    pub fn record_skills(&self, session: Uuid, skills: impl IntoIterator<Item = ActivatedRef>) {
        let skills: Vec<ActivatedRef> = skills.into_iter().collect();
        if skills.is_empty() {
            return;
        }
        if let Ok(mut reg) = self.by_session.lock() {
            merge(&mut reg.entry(session).or_default().skills, skills);
        }
    }

    /// Record the persona the persona stage just activated for `session`.
    pub fn record_persona(&self, session: Uuid, persona: ActivatedRef) {
        if let Ok(mut reg) = self.by_session.lock() {
            merge(&mut reg.entry(session).or_default().personas, [persona]);
        }
    }

    /// What `session` has been given, empty when nothing was recorded.
    pub fn get(&self, session: Uuid) -> SessionActivations {
        self.by_session
            .lock()
            .ok()
            .and_then(|reg| reg.get(&session).cloned())
            .unwrap_or_default()
    }

    /// Forget every session that is no longer live.
    pub fn retain_live(&self, live: &HashSet<Uuid>) {
        if let Ok(mut reg) = self.by_session.lock() {
            reg.retain(|id, _| live.contains(id));
        }
    }
}

fn global() -> &'static ActivationRegistry {
    static REG: OnceLock<ActivationRegistry> = OnceLock::new();
    REG.get_or_init(ActivationRegistry::default)
}

/// Record the skills an activation stage just injected into `session`.
pub fn record_skills(session: Uuid, skills: impl IntoIterator<Item = ActivatedRef>) {
    global().record_skills(session, skills);
}

/// Record the persona the persona stage just activated for `session`.
pub fn record_persona(session: Uuid, persona: ActivatedRef) {
    global().record_persona(session, persona);
}

/// What `session` has been given, empty when nothing was recorded.
pub fn get(session: Uuid) -> SessionActivations {
    global().get(session)
}

/// Forget every session that is no longer live.
pub fn retain_live(live: &HashSet<Uuid>) {
    global().retain_live(live);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::neo4j::traits::*;

    fn r(id: &str) -> ActivatedRef {
        ActivatedRef {
            id: id.into(),
            name: format!("n{id}"),
        }
    }

    #[test]
    fn skills_accumulate_newest_first_without_duplicates() {
        let reg = ActivationRegistry::default();
        let s = Uuid::new_v4();
        reg.record_skills(s, [r("a"), r("b")]);
        reg.record_skills(s, [r("b")]);
        let got = reg.get(s).skills;
        assert_eq!(
            got.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(),
            ["b", "a"]
        );
    }

    #[test]
    fn each_kind_is_bounded() {
        let reg = ActivationRegistry::default();
        let s = Uuid::new_v4();
        reg.record_skills(s, (0..20).map(|i| r(&i.to_string())));
        assert_eq!(reg.get(s).skills.len(), MAX_PER_KIND);
    }

    #[test]
    fn persona_is_kept_apart_from_skills() {
        let reg = ActivationRegistry::default();
        let s = Uuid::new_v4();
        reg.record_skills(s, [r("sk")]);
        reg.record_persona(s, r("pe"));
        let got = reg.get(s);
        assert_eq!(got.skills[0].id, "sk");
        assert_eq!(got.personas[0].id, "pe");
    }

    #[test]
    fn retain_live_forgets_ended_sessions() {
        let reg = ActivationRegistry::default();
        let (live, dead) = (Uuid::new_v4(), Uuid::new_v4());
        reg.record_skills(live, [r("x")]);
        reg.record_skills(dead, [r("y")]);
        reg.retain_live(&HashSet::from([live]));
        assert_eq!(reg.get(live).skills.len(), 1);
        assert!(reg.get(dead).skills.is_empty());
    }

    #[test]
    fn unknown_session_is_empty_not_an_error() {
        let reg = ActivationRegistry::default();
        assert_eq!(reg.get(Uuid::new_v4()), SessionActivations::default());
    }
}
