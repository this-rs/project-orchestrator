//! Conduction — the objective ledger (N1, plan b380137c).
//!
//! One writer per turn (the agent engine's `after_turn`), one reader at session
//! close. The ledger carries the OBJECTIVE a session serves and the state it is
//! in, and it refuses a contradictory verdict: once `Reached`, `Abandoned` or
//! `Blocked`, the objective never returns to `Pursued`.
//!
//! Nothing here is an LLM: the state is settled from the terminal `done` of a
//! turn, and the objective is derived without any model. This module is the
//! state machine only; the proof that a `Reached` is real arrives in N3.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use super::intent::QueryIntentMode;

/// What a session is trying to achieve, derived without an LLM.
#[derive(Debug, Clone)]
pub enum Objective {
    /// A plan drives the session (its task, when known, resolves to [`Objective::Task`]).
    Plan(Uuid),
    /// A specific task, with its title when it has already been read.
    Task {
        task_id: Uuid,
        title: Option<String>,
    },
    /// No task: the declared intent of the first message.
    Intent(QueryIntentMode),
    /// Nothing is declared at all.
    None,
}

/// The state an objective is in, settled once per turn by a single writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveState {
    /// Still being worked on; the only state a new transition may start from.
    Pursued,
    /// Claimed reached. N3 will demand the proof before it is believed.
    Reached,
    /// Given up, with a typed reason (never a free string in production).
    Abandoned { reason: String },
    /// Cannot proceed until the thing named by `on` moves.
    Blocked { on: String },
}

impl ObjectiveState {
    /// A terminal state never regresses: settling it again is fine, anything
    /// else is a contradiction.
    fn is_terminal(&self) -> bool {
        matches!(
            self,
            ObjectiveState::Reached
                | ObjectiveState::Abandoned { .. }
                | ObjectiveState::Blocked { .. }
        )
    }
}

/// A transition of the objective, kept for the record (N7's report).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveTransition {
    pub from: ObjectiveState,
    pub to: ObjectiveState,
    pub at: DateTime<Utc>,
    pub reason: String,
}

/// Why a settle was refused: a terminal state cannot regress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    pub from: ObjectiveState,
    pub to: ObjectiveState,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "objective {:?} cannot settle to {:?}: a terminal state never regresses",
            self.from, self.to
        )
    }
}

/// The ledger of one session's objective. Written by a single owner, read at close.
#[derive(Debug, Clone)]
pub struct ObjectiveLedger {
    objective: Objective,
    state: ObjectiveState,
    transitions: Vec<ObjectiveTransition>,
}

impl ObjectiveLedger {
    pub fn new(objective: Objective) -> Self {
        Self {
            objective,
            state: ObjectiveState::Pursued,
            transitions: Vec::new(),
        }
    }

    pub fn objective(&self) -> &Objective {
        &self.objective
    }

    pub fn state(&self) -> &ObjectiveState {
        &self.state
    }

    pub fn transitions(&self) -> &[ObjectiveTransition] {
        &self.transitions
    }

    /// Settle a new state. Idempotent for the same state; refuses a terminal
    /// state regressing to anything else.
    pub fn settle(
        &mut self,
        to: ObjectiveState,
        at: DateTime<Utc>,
        reason: String,
    ) -> Result<(), Conflict> {
        if self.state == to {
            return Ok(());
        }
        if self.state.is_terminal() {
            return Err(Conflict {
                from: self.state.clone(),
                to,
            });
        }
        let from = self.state.clone();
        self.transitions.push(ObjectiveTransition {
            from: from.clone(),
            to: to.clone(),
            at,
            reason,
        });
        self.state = to;
        Ok(())
    }
}

/// Map a terminal `done` into an objective state (N1 only; N2/N3 refine it).
pub fn state_from_stop(stop_reason: Option<&str>, is_error: bool, subtype: &str) -> ObjectiveState {
    if is_error {
        // A provider fault (a 429, a dead account) is not the objective being abandoned:
        // `turn_reward` already excludes it from the reward, and the objective follows.
        return ObjectiveState::Pursued;
    }
    match stop_reason {
        // The turn limit stopped the work: it resumes once turns are granted.
        Some("max_turns") => ObjectiveState::Blocked {
            on: "turn_limit".into(),
        },
        // The one end `turn_reward` reads as a success.
        Some("completed") => ObjectiveState::Reached,
        // Its second success path: no stop reason, an explicit success subtype.
        None if subtype == "success" => ObjectiveState::Reached,
        // Every other end (interrupt, error, unknown stop) is ambiguous: no verdict.
        _ => ObjectiveState::Pursued,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> DateTime<Utc> {
        Utc::now()
    }

    #[test]
    fn a_new_ledger_is_pursued_and_empty() {
        let ledger = ObjectiveLedger::new(Objective::None);
        assert_eq!(ledger.state(), &ObjectiveState::Pursued);
        assert!(ledger.transitions().is_empty());
    }

    #[test]
    fn pursuing_to_reached_is_recorded() {
        let mut ledger = ObjectiveLedger::new(Objective::Intent(QueryIntentMode::Plan));
        ledger
            .settle(ObjectiveState::Reached, at(), "done".into())
            .unwrap();
        assert_eq!(ledger.state(), &ObjectiveState::Reached);
        assert_eq!(ledger.transitions().len(), 1);
    }

    #[test]
    fn a_terminal_state_never_regresses() {
        let mut ledger = ObjectiveLedger::new(Objective::None);
        let abandoned = ObjectiveState::Abandoned {
            reason: "blocked".into(),
        };
        ledger.settle(abandoned.clone(), at(), "r".into()).unwrap();
        let conflict = ledger
            .settle(ObjectiveState::Reached, at(), "later".into())
            .unwrap_err();
        assert_eq!(conflict.from, abandoned);
        assert_eq!(
            ledger.state(),
            &ObjectiveState::Abandoned {
                reason: "blocked".into()
            }
        );
    }

    #[test]
    fn settling_the_same_state_is_idempotent() {
        let mut ledger = ObjectiveLedger::new(Objective::None);
        let blocked = ObjectiveState::Blocked {
            on: "turn_limit".into(),
        };
        ledger.settle(blocked.clone(), at(), "r".into()).unwrap();
        assert!(ledger.settle(blocked, at(), "r2".into()).is_ok());
        assert_eq!(ledger.transitions().len(), 1);
    }

    #[test]
    fn abandoned_carries_its_reason() {
        let mut ledger = ObjectiveLedger::new(Objective::None);
        ledger
            .settle(
                ObjectiveState::Abandoned {
                    reason: "no progress".into(),
                },
                at(),
                "r".into(),
            )
            .unwrap();
        match ledger.state() {
            ObjectiveState::Abandoned { reason } => assert_eq!(reason, "no progress"),
            other => panic!("expected abandoned, got {:?}", other),
        }
    }

    #[test]
    fn state_from_stop_reads_the_done_like_turn_reward() {
        // The two successes `turn_reward` recognises.
        assert_eq!(
            state_from_stop(Some("completed"), false, ""),
            ObjectiveState::Reached
        );
        assert_eq!(
            state_from_stop(None, false, "success"),
            ObjectiveState::Reached
        );
        // The turn limit stops the work; it resumes once turns are granted.
        assert_eq!(
            state_from_stop(Some("max_turns"), false, ""),
            ObjectiveState::Blocked {
                on: "turn_limit".into()
            }
        );
    }

    #[test]
    fn the_conflict_explains_itself() {
        let conflict = Conflict {
            from: ObjectiveState::Reached,
            to: ObjectiveState::Pursued,
        };
        let text = format!("{conflict}");
        assert!(text.contains("cannot settle"));
    }

    #[test]
    fn the_objective_is_readable() {
        let ledger = ObjectiveLedger::new(Objective::None);
        assert!(matches!(ledger.objective(), Objective::None));
        let transitions: &[ObjectiveTransition] = ledger.transitions();
        assert!(transitions.is_empty());
    }

    #[test]
    fn state_from_stop_treats_an_ambiguous_end_as_no_verdict() {
        // An interrupt can be resumed, an error retried, a provider fault is not the
        // objective being given up: none of them is a verdict.
        assert_eq!(
            state_from_stop(Some("interrupted"), false, ""),
            ObjectiveState::Pursued
        );
        assert_eq!(
            state_from_stop(Some("error"), false, ""),
            ObjectiveState::Pursued
        );
        assert_eq!(
            state_from_stop(None, true, "quota_exhausted"),
            ObjectiveState::Pursued
        );
    }
}
