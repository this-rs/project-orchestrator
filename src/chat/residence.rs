//! Enrichment residence: a block already in the conversation is not re-sent.
//!
//! # The measurement this module answers
//!
//! Measured on 2026-11-28 over the native harness transcripts (plan `b00d80c1`,
//! task GEO-14): 66 % of the cumulative re-read prefix is `user` messages, and
//! 98–100 % of that volume is not what the human typed — it is PO's own
//! enrichment (foundational notes, relevant knowledge, work in progress,
//! reflexes). The cause is not the size of one block but its REPETITION: in one
//! 854-turn session, `Work In Progress` was injected 55 times with only 9
//! distinct bodies, `Foundational Notes` 110 times with 3, the untrusted
//! preamble 55 times with 1. A block written at turn N stays in the prefix and is
//! re-read by every later request, so a 1 417-character block repeated 55 times
//! is paid ~55 times again on every request that follows it.
//!
//! # What this does, and what it does not
//!
//! It rewrites the *projection* of the enrichment for one turn: a section whose
//! body was already delivered in full earlier in this conversation becomes a
//! bounded pointer (identical, or "only the figures moved"). Everything else is
//! delivered in full: a section that changed materially, a section too small for
//! a pointer to pay off, a session's first turn, and any section whose body has
//! not been seen for [`REFRESH_EVERY`] turns (drift, and the earlier message may
//! be gone after a compaction — the ledger is also forgotten on compaction).
//!
//! The durable journal and the graph are untouched: nothing is deleted, nothing
//! is summarized, and the full block is still in the conversation where it was
//! first delivered. Only the bytes this turn adds change.
//!
//! # The two equalities
//!
//! * **identical** — the same body. The untrusted-container opening and closing
//!   lines are removed before comparing: their id is a fresh nonce every turn, so
//!   keeping it would make every turn look new (measured: it is what hid the
//!   repetition).
//! * **only the figures moved** — the same body once digit runs are collapsed, so
//!   `99/110 (90 %)` and `100/110 (91 %)` are one block. Said honestly by the
//!   pointer, never silently.
//!
//! The constants below are mirrored by `scripts/geo_residence_replay.py`, whose
//! `--check` mode replays real transcripts; the test
//! [`tests::policy_matches_the_replay_script`] refuses a divergence.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use uuid::Uuid;

use super::enrichment::EnrichmentSection;

/// Below this size, a pointer costs nearly as much as the block: deliver it.
pub const SECTION_FLOOR: usize = 240;

/// A body that has not been delivered in full for this many turns is delivered
/// again, whatever the ledger says (drift, and a compaction may have removed the
/// earlier message).
pub const REFRESH_EVERY: u32 = 40;

/// How many distinct bodies one title may remember (bounded memory per session).
const MAX_BODIES: usize = 32;

/// The per-session residence ledger. One per chat manager; a session's entries
/// live from its first turn to its close (or its next compaction).
#[derive(Debug, Default)]
pub struct ResidenceLedger {
    sessions: Mutex<HashMap<Uuid, SessionLedger>>,
}

#[derive(Debug, Default)]
struct SessionLedger {
    /// Turns this session has had enrichment applied for (1-based).
    turn: u32,
    titles: HashMap<String, TitleState>,
}

#[derive(Debug, Default)]
struct TitleState {
    /// body fingerprint → the turn it was first delivered in full
    exact: HashMap<u64, u32>,
    /// digit-collapsed fingerprint → the turn it was first delivered in full
    loose: HashMap<u64, u32>,
    /// the turn of the last FULL delivery of this title, whatever the body
    last_full: Option<u32>,
}

/// What one application did, for the log line and for the tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidenceOutcome {
    /// Sections the pipeline produced.
    pub sections: usize,
    /// Sections delivered in full.
    pub delivered: usize,
    /// Sections replaced by a bounded pointer.
    pub pointers: usize,
    /// Characters this turn did not add, thanks to pointers.
    pub chars_saved: usize,
}

/// The process-wide ledger. A conversation belongs to one process, and every
/// manager of that process sees the same conversation: one ledger, not one per
/// manager, or a session driven by two engines would point at messages the other
/// never delivered.
pub fn global() -> Arc<ResidenceLedger> {
    static GLOBAL: OnceLock<Arc<ResidenceLedger>> = OnceLock::new();
    GLOBAL.get_or_init(|| Arc::new(ResidenceLedger::new())).clone()
}

impl ResidenceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// The engine opens a turn: the clock moves once, here.
    ///
    /// Every projection of that turn ([`Self::apply`]) then uses the same number,
    /// so a prompt recomputed within one turn can never point at a message the
    /// conversation never received, and a real next turn can. Called by the turn
    /// entry point of each engine (`AgentRuntime::open_turn` and
    /// `ChatManager::stream_response`) — the two places that know a turn began.
    pub fn begin_turn(&self, session: Uuid) -> u32 {
        let mut guard = self
            .sessions
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let ledger = guard.entry(session).or_default();
        ledger.turn = ledger.turn.saturating_add(1);
        ledger.turn
    }

    /// Apply the residence policy to the current turn's sections, in place.
    ///
    /// Idempotent within a turn: the clock is advanced by [`Self::begin_turn`],
    /// not here, so a second projection of the same turn sees the bodies it just
    /// delivered and delivers them again (a body is only turned into a pointer
    /// when it was delivered at an EARLIER turn). A session whose turn was never
    /// opened opens its first one here, so a caller that forgets to is merely
    /// less economical, never wrong.
    pub fn apply(&self, session: Uuid, sections: &mut [EnrichmentSection]) -> ResidenceOutcome {
        let mut outcome = ResidenceOutcome {
            sections: sections.len(),
            ..Default::default()
        };
        if sections.is_empty() {
            return outcome;
        }
        let mut guard = self
            .sessions
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let ledger = guard.entry(session).or_default();
        if ledger.turn == 0 {
            ledger.turn = 1;
        }
        let turn = ledger.turn;

        for section in sections.iter_mut() {
            let size = section.content.len();
            if size < SECTION_FLOOR {
                outcome.delivered += 1;
                continue;
            }
            let body = body_of(&section.content);
            if body.is_empty() {
                outcome.delivered += 1;
                continue;
            }
            let exact = fingerprint(&section.title, &body);
            let loose = fingerprint(&section.title, &collapse_digits(&body));

            let state = ledger.titles.entry(section.title.clone()).or_default();
            let exact_seen = state.exact.get(&exact).copied();
            let loose_seen = state.loose.get(&loose).copied();
            let stale = match state.last_full {
                None => true,
                Some(at) => turn.saturating_sub(at) >= REFRESH_EVERY,
            };

            let earlier = |seen: Option<u32>| seen.filter(|at| *at < turn);
            let pointer = match (earlier(exact_seen), earlier(loose_seen), stale) {
                (_, _, true) => None,
                (Some(at), _, false) => Some(pointer_identical(at, size)),
                (None, Some(at), false) => Some(pointer_figures(at, size)),
                (None, None, false) => None,
            };

            match pointer {
                Some(text) => {
                    outcome.pointers += 1;
                    outcome.chars_saved += size.saturating_sub(text.len());
                    section.content = text;
                }
                None => {
                    outcome.delivered += 1;
                    state.last_full = Some(turn);
                }
            }
            remember(&mut state.exact, exact, turn);
            remember(&mut state.loose, loose, turn);
        }
        outcome
    }

    /// Forget a session: its next turn delivers everything in full again. Called
    /// when the session closes, and after a compaction (the earlier messages the
    /// pointers refer to may be gone).
    pub fn forget(&self, session: Uuid) {
        self.sessions
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(&session);
    }

    /// Turns recorded for a session (0 before its first enriched turn).
    pub fn turn_of(&self, session: Uuid) -> u32 {
        self.sessions
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&session)
            .map(|s| s.turn)
            .unwrap_or(0)
    }
}

/// Record a fingerprint, dropping the oldest entries past [`MAX_BODIES`].
fn remember(map: &mut HashMap<u64, u32>, key: u64, turn: u32) {
    map.entry(key).or_insert(turn);
    while map.len() > MAX_BODIES {
        let oldest = map
            .iter()
            .min_by_key(|(_, at)| **at)
            .map(|(k, _)| *k);
        match oldest {
            Some(k) => {
                map.remove(&k);
            }
            None => break,
        }
    }
}

/// The section body without its untrusted container lines.
///
/// The container is re-opened with a fresh random id every turn
/// (`chat::untrusted::wrap_random`), so comparing raw content would make every
/// turn look new — measured, this is what hid the repetition.
fn body_of(content: &str) -> String {
    content
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !(t.starts_with("<untrusted_data") || t.starts_with("</untrusted_data"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Digits collapsed and whitespace squeezed: progress counters are not a change
/// of substance (`99/110 (90 %)` → `#/# (# %)`).
fn collapse_digits(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    let mut in_digits = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            if !in_digits {
                out.push('#');
                in_digits = true;
                last_space = false;
            }
            continue;
        }
        in_digits = false;
        if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    out.trim().to_string()
}

/// FNV-1a over the title and the normalized body: stable across processes, so a
/// test can assert on a fingerprint.
fn fingerprint(title: &str, body: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in title
        .as_bytes()
        .iter()
        .chain(b"\0".iter())
        .chain(body.as_bytes().iter())
    {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// The pointer for a body already delivered verbatim at `turn`.
pub fn pointer_identical(turn: u32, chars: usize) -> String {
    format!(
        "_[unchanged since turn {turn}: this block is already in an earlier message \
of this conversation ({chars} chars not repeated)]_"
    )
}

/// The pointer for a body whose only difference is its figures.
pub fn pointer_figures(turn: u32, chars: usize) -> String {
    format!(
        "_[same block as turn {turn}, only figures changed (progress counters): \
not repeated, see the earlier message ({chars} chars)]_"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::enrichment::EnrichmentSource;

    fn section(title: &str, content: impl Into<String>) -> EnrichmentSection {
        EnrichmentSection {
            title: title.to_string(),
            content: content.into(),
            source: "test".to_string(),
            enrichment_source: EnrichmentSource::StatusInjection,
        }
    }

    /// A real capture shape: the graph wraps a section in a container whose id is
    /// a fresh nonce every turn.
    fn wrapped(nonce: &str, body: &str) -> String {
        format!(
            "<untrusted_data id=\"{nonce}\" source=\"plan_status\" project=\"nexus-sdk\">\n{body}\n</untrusted_data id=\"{nonce}\">"
        )
    }

    fn wip() -> String {
        let mut s = String::from("- **Plan:** Parité totale Claude Code ↔ harnais natif (priority: 95)\n  Progress: 99/110  (90%)\n");
        for i in 0..20 {
            s.push_str(&format!("  🔄 in_progress: TÂCHE-{i} — un intitulé de tâche assez long pour peser\n"));
        }
        s
    }

    #[test]
    fn an_identical_block_becomes_a_pointer_at_the_next_turn() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();

        let first = vec![section("Work In Progress", wrapped("a1b2", &body))];
        let mut first = first;
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut first);
        assert_eq!(out.delivered, 1);
        assert_eq!(out.pointers, 0);
        assert!(first[0].content.contains("TÂCHE-0"), "delivered in full");

        // Same body, new nonce: the repetition the container id used to hide.
        let mut second = vec![section("Work In Progress", wrapped("ffff", &body))];
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut second);
        assert_eq!(out.pointers, 1);
        assert_eq!(out.delivered, 0);
        assert!(
            second[0].content.contains("unchanged since turn 1"),
            "the pointer names the turn it was delivered at: {}",
            second[0].content
        );
        assert!(
            !second[0].content.contains("TÂCHE-0"),
            "the block's text is not repeated"
        );
        assert!(out.chars_saved > 1_000, "saved {}", out.chars_saved);
    }

    #[test]
    fn a_material_change_is_delivered_in_full() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut a = vec![section("Work In Progress", wrapped("a1", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut a);

        let changed = body.replace("TÂCHE-3", "TÂCHE-3 (terminée)").replace("🔄", "✅");
        let mut b = vec![section("Work In Progress", wrapped("a2", &changed))];
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut b);
        assert_eq!(out.delivered, 1);
        assert_eq!(out.pointers, 0);
        assert!(b[0].content.contains("✅"));
    }

    #[test]
    fn figures_alone_are_a_honest_pointer_not_a_repetition() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut a = vec![section("Work In Progress", wrapped("a1", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut a);

        let counted = body.replace("99/110  (90%)", "100/110  (91%)");
        assert_ne!(body, counted);
        let mut b = vec![section("Work In Progress", wrapped("a2", &counted))];
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut b);
        assert_eq!(out.pointers, 1);
        assert!(
            b[0].content.contains("only figures changed"),
            "the pointer says what differs: {}",
            b[0].content
        );
    }

    #[test]
    fn a_small_block_is_always_delivered() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let small = "L4 (L4 — Méta-cognitif) — competence: 94%, pain: 43%";
        assert!(small.len() < SECTION_FLOOR);
        let mut a = vec![section("Scaffolding Level", small)];
        let mut b = vec![section("Scaffolding Level", small)];
        ledger.begin_turn(session);
        ledger.apply(session, &mut a);
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut b);
        assert_eq!(out.pointers, 0, "a pointer would cost as much as the block");
        assert_eq!(b[0].content, small);
    }

    #[test]
    fn an_older_snapshot_coming_back_is_still_a_repeat() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let a_body = wip();
        let b_body = wip().replace("TÂCHE-1", "TÂCHE-1 (bloquée)");

        let mut first = vec![section("Work In Progress", wrapped("n1", &a_body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut first);
        let mut second = vec![section("Work In Progress", wrapped("n2", &b_body))];
        ledger.begin_turn(session);
        assert_eq!(ledger.apply(session, &mut second).delivered, 1);
        let mut third = vec![section("Work In Progress", wrapped("n3", &a_body))];
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut third);
        assert_eq!(out.pointers, 1, "the first snapshot is already in the prefix");
        assert!(third[0].content.contains("unchanged since turn 1"));
    }

    #[test]
    fn a_stale_block_is_delivered_again() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut first = vec![section("Work In Progress", wrapped("n0", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut first);
        // Walk the session forward without ever delivering this title in full.
        let other = "x".repeat(SECTION_FLOOR + 10);
        for _ in 0..REFRESH_EVERY {
            let mut filler = vec![section("Reflexes", format!("{other}{}", Uuid::new_v4()))];
            ledger.begin_turn(session);
            ledger.apply(session, &mut filler);
        }
        let mut again = vec![section("Work In Progress", wrapped("n1", &body))];
        ledger.begin_turn(session);
        let out = ledger.apply(session, &mut again);
        assert_eq!(out.delivered, 1, "refresh after {REFRESH_EVERY} turns");
        assert!(again[0].content.contains("TÂCHE-0"));
    }

    #[test]
    fn forgetting_a_session_delivers_everything_again() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut a = vec![section("Work In Progress", wrapped("n0", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut a);
        ledger.forget(session);
        assert_eq!(ledger.turn_of(session), 0);
        let mut b = vec![section("Work In Progress", wrapped("n1", &body))];
        ledger.begin_turn(session);
        assert_eq!(ledger.apply(session, &mut b).delivered, 1);
    }

    #[test]
    fn applying_twice_in_one_turn_projects_the_same_bytes() {
        // A turn whose prompt is recomputed (a retry) must not eat its own block.
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut older = vec![section("Reflexes", format!("{}{}", "z".repeat(300), Uuid::nil()))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut older);

        let mut first = vec![section("Work In Progress", wrapped("n0", &body))];
        ledger.begin_turn(session);
        let out1 = ledger.apply(session, &mut first);
        // The retry: the same turn, its prompt projected again. No `begin_turn`,
        // because the conversation has not had a turn since — a pointer here would
        // name a message the provider never received.
        let mut second = vec![section("Work In Progress", wrapped("n1", &body))];
        let out2 = ledger.apply(session, &mut second);
        assert_eq!(out1, out2, "the same turn projects the same thing twice");
        assert!(first[0].content.contains("TÂCHE-0"));
        assert!(second[0].content.contains("TÂCHE-0"));
    }

    #[test]
    fn a_pointer_carries_no_graph_text_and_no_container() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut a = vec![section("Work In Progress", wrapped("n0", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut a);
        let mut b = vec![section("Work In Progress", wrapped("n1", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut b);
        assert!(!b[0].content.contains("<untrusted_data"), "no container");
        assert!(!b[0].content.contains("TÂCHE-"), "no graph text");
        assert!(
            b[0].content.len() < 200,
            "a pointer stays bounded: {} chars",
            b[0].content.len()
        );
    }

    #[test]
    fn titles_are_kept_apart() {
        let ledger = ResidenceLedger::new();
        let session = Uuid::new_v4();
        let body = wip();
        let mut a = vec![section("Work In Progress", wrapped("n0", &body))];
        ledger.begin_turn(session);
        ledger.apply(session, &mut a);
        let mut b = vec![section("Relevant Notes", wrapped("n1", &body))];
        ledger.begin_turn(session);
        assert_eq!(
            ledger.apply(session, &mut b).delivered,
            1,
            "the same text under another title is another block"
        );
    }

    #[test]
    fn policy_matches_the_replay_script() {
        // The script replays real transcripts with these numbers; a silent change on
        // one side would make its --check meaningless.
        let script = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/geo_residence_replay.py"),
        )
        .expect("scripts/geo_residence_replay.py is in the repository");
        assert!(
            script.contains(&format!("SECTION_FLOOR = {SECTION_FLOOR}")),
            "SECTION_FLOOR diverged"
        );
        assert!(
            script.contains(&format!("REFRESH_EVERY = {REFRESH_EVERY}")),
            "REFRESH_EVERY diverged"
        );
    }
}
