//! The retrieval trigger: turning « the index was never asked » into an INVITATION.
//!
//! Why this exists (formalisme v7): with a perfect index, the measured share of MISSES
//! stays at 100 % as long as `p_trig = 0` — nobody retrieves, so nothing is mobilised.
//! The missing link is not the content, it is the question. This module decides WHEN to
//! ask, and never WHAT: the invitation is one short line added to the USER turn (never
//! to the system prompt, whose prefix must stay byte-identical for the cache), and it
//! carries no identifier — the retrieval that follows is the agent's own call.
//!
//! Signals implemented, each with its own witness:
//!
//! - **task boundary** — the session's first turn, or a turn that names a work item
//!   (`#task:` / `#plan:` / `#milestone:` / `#release:`): the agent enters a unit of
//!   work, which is exactly where the graph knows something the turn does not;
//! - **declared uncertainty** — the message says it does not know, or has no source.
//!
//! - **contradiction with the core** — a retrieved piece that is no longer current
//!   knowledge (superseded, archived, out of its validity window). It is not a
//!   message-based signal, so it is NOT detected here: the injection stage
//!   (`knowledge_injection`) records it to the ledger under
//!   [`Trigger::Contradiction`] with the stale note as its witness — a witness
//!   that only PR5 (validity windows + `supersedes`) could provide.
//!
//! The flag is **off** by default: `RETRIEVAL_TRIGGER=1` turns it on. Off, the manager
//! builds the turn exactly as before.

use crate::notes::retrieval_ledger::Trigger;

/// The environment variable that turns the trigger on. Absent, `0`, `false` or `no`:
/// off, and the turn is built exactly as before this module existed.
pub const FLAG: &str = "RETRIEVAL_TRIGGER";

/// What the turn knows, and nothing else: the trigger reads no graph, does no I/O.
#[derive(Clone, Copy, Debug)]
pub struct Facts<'a> {
    /// The text the user typed (attachments' text included, references excluded).
    pub message: &'a str,
    /// The session carries no answered turn yet (`message_count <= 1`): the agent is
    /// opening a unit of work.
    pub first_turn: bool,
}

/// Whether the trigger is on for this process.
pub fn enabled() -> bool {
    enabled_from(std::env::var(FLAG).ok().as_deref())
}

fn enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// The signals of the message, in the order they are tested: a turn that opens a unit of
/// work is a boundary even if it also says it does not know — the boundary explains the
/// miss better than the uncertainty does.
pub fn detect(facts: &Facts<'_>) -> Option<Trigger> {
    if facts.first_turn {
        return Some(Trigger::TaskBoundary);
    }
    if names_a_work_item(facts.message) {
        return Some(Trigger::TaskBoundary);
    }
    if declares_uncertainty(facts.message) {
        return Some(Trigger::Uncertainty);
    }
    None
}

/// The invite to add to the user turn, or `None` when the flag is off or no signal fires.
pub fn for_turn(facts: Facts<'_>) -> Option<String> {
    for_turn_with(enabled(), facts)
}

fn for_turn_with(on: bool, facts: Facts<'_>) -> Option<String> {
    if !on {
        return None;
    }
    detect(&facts).map(invitation)
}

/// The one line the agent reads. It names its motive, asks a question, and states that
/// nothing has been retrieved yet — so the turn cannot mistake the invitation for a
/// result (the mistake a silent injection makes).
pub fn invitation(signal: Trigger) -> String {
    format!(
        "**Récupération** (motif : {}) — avant de répondre, interroge le graphe : \
         `note(action: \"search_semantic\")`, `decision(action: \"search_semantic\")`, ou \
         l'outil de recherche du projet. Ceci est une invitation, pas un résultat : aucune \
         mémoire n'a été mobilisée pour ce tour.",
        motive(signal)
    )
}

/// The French label of a signal, for the invitation and the journal.
pub fn motive(signal: Trigger) -> &'static str {
    match signal {
        Trigger::TaskBoundary => "frontière de tâche",
        Trigger::Uncertainty => "incertitude déclarée",
        Trigger::Contradiction => "contradiction avec le noyau",
        Trigger::Explicit => "explicite",
        Trigger::Hook => "hook",
        Trigger::Injection => "injection",
        Trigger::System => "système",
    }
}

/// The work items of the product: a turn that names one enters a unit of work.
const WORK_ITEM_MARKERS: [&str; 4] = ["#task:", "#plan:", "#milestone:", "#release:"];

fn names_a_work_item(message: &str) -> bool {
    WORK_ITEM_MARKERS
        .iter()
        .any(|m| message.to_ascii_lowercase().contains(m))
}

/// What a turn says when the answer is not in the turn. Kept short and literal: a
/// phrase in this list is a declaration, not a guess about intent.
const UNCERTAINTY_MARKERS: [&str; 18] = [
    "je ne sais pas",
    "je n'en sais rien",
    "je ne suis pas sûr",
    "je ne suis pas certain",
    "je n'ai pas l'information",
    "je n'ai pas cette information",
    "je n'ai pas la réponse",
    "je ne trouve pas",
    "je ne vois pas",
    "je n'ai pas accès",
    "sans source",
    "aucune source",
    "pas de source",
    "i don't know",
    "i do not know",
    "not sure",
    "no idea",
    "cannot find",
];

fn declares_uncertainty(message: &str) -> bool {
    let m = message.to_lowercase();
    UNCERTAINTY_MARKERS.iter().any(|p| m.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(message: &str, first_turn: bool) -> Facts<'_> {
        Facts {
            message,
            first_turn,
        }
    }

    #[test]
    fn the_first_turn_of_a_session_is_a_task_boundary() {
        assert_eq!(detect(&facts("bonjour", true)), Some(Trigger::TaskBoundary));
    }

    #[test]
    fn a_turn_that_names_a_work_item_is_a_task_boundary() {
        for message in [
            "on attaque #task:0d10391a-525c-42f6-8e14-384150ff7048",
            "reprends le #plan:42d49a31 et dis-moi où on en est",
            "ce #milestone:9f2c est en retard",
            "#release:0.2.0 doit sortir",
        ] {
            assert_eq!(
                detect(&facts(message, false)),
                Some(Trigger::TaskBoundary),
                "{message}"
            );
        }
    }

    #[test]
    fn an_ordinary_continuation_fires_nothing() {
        for message in ["continue", "merci", "et ensuite ?", "relance les tests"] {
            assert_eq!(detect(&facts(message, false)), None, "{message}");
        }
    }

    #[test]
    fn a_declared_uncertainty_fires_each_time_it_is_declared() {
        for message in [
            "je ne sais pas où est défini ce handler",
            "Je ne trouve pas la note sur le pool Neo4j",
            "aucune source sur ce point",
            "I don't know which crate owns that",
            "not sure about the retry policy",
        ] {
            assert_eq!(
                detect(&facts(message, false)),
                Some(Trigger::Uncertainty),
                "{message}"
            );
        }
    }

    #[test]
    fn a_boundary_wins_over_an_uncertainty_in_the_same_turn() {
        assert_eq!(
            detect(&facts("je ne sais pas quoi faire de #task:abc", false)),
            Some(Trigger::TaskBoundary)
        );
    }

    #[test]
    fn the_flag_is_off_by_default() {
        assert!(!enabled_from(None));
        for off in ["", "0", "false", "no", "off", "  "] {
            assert!(!enabled_from(Some(off)), "{off:?}");
        }
        for on in ["1", "true", "TRUE", "yes", " on "] {
            assert!(enabled_from(Some(on)), "{on:?}");
        }
    }

    #[test]
    fn off_the_turn_is_built_exactly_as_before() {
        assert_eq!(
            for_turn_with(false, facts("je ne sais pas", true)),
            None,
            "a flag at false must add nothing"
        );
    }

    #[test]
    fn on_the_invitation_names_its_motive_and_promises_nothing() {
        let text = for_turn_with(true, facts("je ne sais pas", false)).expect("a signal fired");
        assert!(text.contains("invitation"), "{text}");
        assert!(text.contains("incertitude déclarée"), "{text}");
        assert!(
            text.contains("aucune mémoire n'a été mobilisée"),
            "an invitation must not pass for a result: {text}"
        );
        // No identifier: what is retrieved, and whether anything is, is the agent's call.
        assert!(!text.contains("#note:"), "{text}");
        assert!(!text.contains("#decision:"), "{text}");
        assert!(!text.contains("uuid"), "{text}");
    }

    #[test]
    fn on_a_boundary_the_invitation_says_so() {
        let text = for_turn_with(true, facts("bonjour", true)).expect("a signal fired");
        assert!(text.contains("frontière de tâche"), "{text}");
        assert!(!text.contains("incertitude"), "{text}");
    }

    #[test]
    fn every_signal_has_its_own_motive() {
        let mut seen = std::collections::HashSet::new();
        for &signal in crate::notes::retrieval_ledger::triggers() {
            assert!(!motive(signal).is_empty());
            assert!(seen.insert(motive(signal)), "{signal:?} shares its label");
        }
    }
}
