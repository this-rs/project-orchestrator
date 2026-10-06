//! What the server is doing while it starts.
//!
//! `start_server` runs a long sequence (bind the port, connect the databases, start the
//! watchers, recover what a previous run left behind, ...) and nothing outside could tell where it
//! was: the desktop splash waited for `/health` and knew only "not yet". The [`BootTracker`] is the
//! shared, readable record of that sequence: named phases, each pending, running, done, failed or
//! skipped, with a duration and, when it makes sense, a counter (`3 / 12 batches`).
//!
//! It is a *record*, never a gate: no phase waits for another because of it, and a failure to
//! record (a poisoned lock) never fails the startup.
//!
//! Details are redacted on the way in ([`redact`]): a snapshot is meant to be shown to the user and
//! one day served over HTTP, so a connection error that carries `bolt://user:secret@host` must not
//! leak the secret.

use std::fmt::Display;
use std::future::Future;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Where a phase stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    /// Declared, not started.
    Pending,
    /// Started, not finished.
    Running,
    /// Finished normally.
    Done,
    /// Finished with an error. Not necessarily fatal: most phases of the startup carry on.
    Failed,
    /// Not run, on purpose (not configured, nothing to do) or never reached.
    Skipped,
}

/// One named step of the startup, as seen from outside.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Phase {
    pub id: String,
    pub label: String,
    pub status: PhaseStatus,
    /// Work done so far, for a phase that counts (`processed` of `total`).
    pub done: Option<u64>,
    pub total: Option<u64>,
    /// Why it failed or was skipped, or a short remark (redacted).
    pub detail: Option<String>,
    /// How long it has run (while running) or took (once finished); absent before it starts.
    pub elapsed_ms: Option<u64>,
}

/// Everything a reader needs, at one instant.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BootSnapshot {
    /// Version of this build.
    pub version: String,
    /// When the server started, in milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// Since then.
    pub elapsed_ms: u64,
    /// The startup sequence is over (the server listens, or the setup server does).
    pub finished: bool,
    pub phases: Vec<Phase>,
}

impl BootSnapshot {
    /// Phases that failed.
    pub fn failed(&self) -> impl Iterator<Item = &Phase> {
        self.phases
            .iter()
            .filter(|p| p.status == PhaseStatus::Failed)
    }
}

#[derive(Debug)]
struct Slot {
    phase: Phase,
    started: Option<Instant>,
    ended: Option<Instant>,
}

#[derive(Debug)]
struct Inner {
    started: Instant,
    started_unix_ms: u64,
    finished: bool,
    slots: Vec<Slot>,
}

/// The shared record. One lives for the process ([`tracker`]); tests build their own.
#[derive(Debug)]
pub struct BootTracker {
    inner: Mutex<Inner>,
}

impl Default for BootTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BootTracker {
    /// An empty tracker, started now.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                started: Instant::now(),
                started_unix_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
                finished: false,
                slots: Vec::new(),
            }),
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        // A panic elsewhere must not turn the record into a second failure.
        f(&mut self.inner.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn slot<'a>(inner: &'a mut Inner, id: &str) -> &'a mut Slot {
        if let Some(at) = inner.slots.iter().position(|s| s.phase.id == id) {
            return &mut inner.slots[at];
        }
        // A phase that was not declared is still recorded, at the end, labelled by its id.
        inner.slots.push(Slot {
            phase: Phase {
                id: id.to_owned(),
                label: id.to_owned(),
                status: PhaseStatus::Pending,
                done: None,
                total: None,
                detail: None,
                elapsed_ms: None,
            },
            started: None,
            ended: None,
        });
        let last = inner.slots.len() - 1;
        &mut inner.slots[last]
    }

    /// Announces the phases the startup will go through, in order, so a reader can show the
    /// ones still to come. Declaring again is harmless: order is kept, a label is updated.
    pub fn declare(&self, phases: &[(&str, &str)]) {
        self.with(|inner| {
            for (id, label) in phases {
                let slot = Self::slot(inner, id);
                slot.phase.label = (*label).to_owned();
            }
        });
    }

    /// The phase begins.
    pub fn start(&self, id: &str) {
        self.with(|inner| {
            let slot = Self::slot(inner, id);
            slot.phase.status = PhaseStatus::Running;
            slot.started = Some(Instant::now());
            slot.ended = None;
            slot.phase.detail = None;
        });
    }

    /// `done` of `total` units of work. Starts the phase if it had not started.
    pub fn progress(&self, id: &str, done: u64, total: u64) {
        self.with(|inner| {
            let slot = Self::slot(inner, id);
            if slot.phase.status == PhaseStatus::Pending {
                slot.phase.status = PhaseStatus::Running;
                slot.started = Some(Instant::now());
            }
            slot.phase.done = Some(done);
            slot.phase.total = Some(total);
        });
    }

    /// A short remark about the phase while it runs or after it ends.
    pub fn detail(&self, id: &str, text: impl Display) {
        let text = redact(&text.to_string());
        self.with(|inner| Self::slot(inner, id).phase.detail = Some(text));
    }

    /// The phase ended normally.
    pub fn done(&self, id: &str) {
        self.end(id, PhaseStatus::Done, None);
    }

    /// The phase ended normally, with a remark ("local-only mode", "3 runs recovered").
    pub fn done_with(&self, id: &str, detail: impl Display) {
        self.end(id, PhaseStatus::Done, Some(redact(&detail.to_string())));
    }

    /// The phase ended with an error. Redacted: see the module documentation.
    pub fn fail(&self, id: &str, error: impl Display) {
        self.end(id, PhaseStatus::Failed, Some(redact(&error.to_string())));
    }

    /// The phase was not run, with the reason.
    pub fn skip(&self, id: &str, reason: impl Display) {
        self.end(id, PhaseStatus::Skipped, Some(redact(&reason.to_string())));
    }

    fn end(&self, id: &str, status: PhaseStatus, detail: Option<String>) {
        self.with(|inner| {
            let slot = Self::slot(inner, id);
            slot.phase.status = status;
            slot.ended = Some(Instant::now());
            if detail.is_some() {
                slot.phase.detail = detail;
            }
        });
    }

    /// Runs `work` as the phase `id`: started before, done after `Ok`, failed after `Err`. The
    /// result is returned untouched.
    pub async fn run<T, E: Display>(
        &self,
        id: &str,
        work: impl Future<Output = Result<T, E>>,
    ) -> Result<T, E> {
        self.start(id);
        let result = work.await;
        match &result {
            Ok(_) => self.done(id),
            Err(error) => self.fail(id, error),
        }
        result
    }

    /// The startup sequence is over. Phases never reached are marked skipped, so a reader never
    /// waits for one that will not come.
    pub fn finish(&self) {
        self.with(|inner| {
            for slot in &mut inner.slots {
                if slot.phase.status == PhaseStatus::Pending {
                    slot.phase.status = PhaseStatus::Skipped;
                    slot.phase.detail = Some("not reached".to_owned());
                }
            }
            inner.finished = true;
        });
    }

    /// A copy of the record, readable from any thread at any time.
    pub fn snapshot(&self) -> BootSnapshot {
        self.with(|inner| {
            let now = Instant::now();
            let ms = |d: std::time::Duration| u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
            BootSnapshot {
                version: env!("CARGO_PKG_VERSION").to_owned(),
                started_unix_ms: inner.started_unix_ms,
                elapsed_ms: ms(now.duration_since(inner.started)),
                finished: inner.finished,
                phases: inner
                    .slots
                    .iter()
                    .map(|slot| {
                        let mut phase = slot.phase.clone();
                        phase.elapsed_ms = slot
                            .started
                            .map(|start| ms(slot.ended.unwrap_or(now).duration_since(start)));
                        phase
                    })
                    .collect(),
            }
        })
    }
}

/// The record of this process.
pub fn tracker() -> &'static BootTracker {
    static TRACKER: OnceLock<BootTracker> = OnceLock::new();
    TRACKER.get_or_init(BootTracker::new)
}

/// Removes what must never reach a screen or a log from a message: the user and password of any
/// `scheme://user:password@host` it contains.
pub fn redact(message: &str) -> String {
    let mut out = String::with_capacity(message.len());
    let mut rest = message;
    while let Some(at) = rest.find("://") {
        let (head, tail) = rest.split_at(at + 3);
        out.push_str(head);
        // The authority ends at the first `/`, whitespace or quote.
        let end = tail
            .find(|c: char| c == '/' || c.is_whitespace() || c == '"' || c == '\'')
            .unwrap_or(tail.len());
        let (authority, after) = tail.split_at(end);
        match authority.rfind('@') {
            Some(at_sign) => {
                out.push_str("***@");
                out.push_str(&authority[at_sign + 1..]);
            }
            None => out.push_str(authority),
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase<'a>(snapshot: &'a BootSnapshot, id: &str) -> &'a Phase {
        snapshot.phases.iter().find(|p| p.id == id).unwrap()
    }

    #[test]
    fn a_phase_goes_pending_running_done_with_a_duration() {
        let boot = BootTracker::new();
        boot.declare(&[("db", "Connecting the databases")]);
        let before = boot.snapshot();
        assert_eq!(phase(&before, "db").status, PhaseStatus::Pending);
        assert_eq!(
            phase(&before, "db").elapsed_ms,
            None,
            "not started: no duration"
        );

        boot.start("db");
        let running = boot.snapshot();
        assert_eq!(phase(&running, "db").status, PhaseStatus::Running);
        assert!(phase(&running, "db").elapsed_ms.is_some());

        std::thread::sleep(std::time::Duration::from_millis(15));
        boot.done("db");
        let done = boot.snapshot();
        assert_eq!(phase(&done, "db").status, PhaseStatus::Done);
        let took = phase(&done, "db").elapsed_ms.unwrap();
        assert!(took >= 15, "{took} ms");
        // Once done, the duration stops growing.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(boot.snapshot().phases[0].elapsed_ms, Some(took));
    }

    #[test]
    fn failed_and_skipped_carry_their_reason() {
        let boot = BootTracker::new();
        boot.start("nats");
        boot.fail("nats", "connection refused");
        boot.skip("watchers", "disabled");
        let s = boot.snapshot();
        assert_eq!(phase(&s, "nats").status, PhaseStatus::Failed);
        assert_eq!(
            phase(&s, "nats").detail.as_deref(),
            Some("connection refused")
        );
        assert_eq!(phase(&s, "watchers").status, PhaseStatus::Skipped);
        assert_eq!(s.failed().count(), 1);
    }

    #[test]
    fn declared_order_is_kept_and_declaring_twice_changes_nothing_but_labels() {
        let boot = BootTracker::new();
        boot.declare(&[("a", "A"), ("b", "B"), ("c", "C")]);
        boot.start("b");
        boot.declare(&[("c", "C renamed"), ("a", "A"), ("d", "D")]);
        let ids: Vec<_> = boot
            .snapshot()
            .phases
            .iter()
            .map(|p| p.id.clone())
            .collect();
        assert_eq!(ids, ["a", "b", "c", "d"]);
        let s = boot.snapshot();
        assert_eq!(phase(&s, "c").label, "C renamed");
        assert_eq!(
            phase(&s, "b").status,
            PhaseStatus::Running,
            "a re-declaration never resets a phase"
        );
    }

    #[test]
    fn a_phase_that_was_not_declared_is_still_recorded_at_the_end() {
        let boot = BootTracker::new();
        boot.declare(&[("a", "A")]);
        boot.start("surprise");
        let s = boot.snapshot();
        assert_eq!(s.phases.last().unwrap().id, "surprise");
        assert_eq!(s.phases.last().unwrap().label, "surprise");
    }

    #[test]
    fn progress_counts_and_starts_the_phase() {
        let boot = BootTracker::new();
        boot.declare(&[("migrations", "Data migrations")]);
        boot.progress("migrations", 3, 12);
        let s = boot.snapshot();
        let m = phase(&s, "migrations");
        assert_eq!(
            (m.status, m.done, m.total),
            (PhaseStatus::Running, Some(3), Some(12))
        );
        boot.progress("migrations", 12, 12);
        boot.done("migrations");
        let s = boot.snapshot();
        assert_eq!(phase(&s, "migrations").done, Some(12));
        assert_eq!(phase(&s, "migrations").status, PhaseStatus::Done);
    }

    #[test]
    fn finishing_marks_what_was_never_reached_so_nobody_waits_for_it() {
        let boot = BootTracker::new();
        boot.declare(&[("a", "A"), ("b", "B")]);
        boot.start("a");
        boot.done("a");
        assert!(!boot.snapshot().finished);
        boot.finish();
        let s = boot.snapshot();
        assert!(s.finished);
        assert_eq!(phase(&s, "a").status, PhaseStatus::Done);
        assert_eq!(phase(&s, "b").status, PhaseStatus::Skipped);
        assert_eq!(phase(&s, "b").detail.as_deref(), Some("not reached"));
    }

    #[tokio::test]
    async fn run_records_the_outcome_and_returns_the_result_untouched() {
        let boot = BootTracker::new();
        let ok: Result<u32, String> = boot.run("good", async { Ok(7) }).await;
        assert_eq!(ok, Ok(7));
        let err: Result<u32, String> = boot.run("bad", async { Err("boom".to_owned()) }).await;
        assert_eq!(err, Err("boom".to_owned()));
        let s = boot.snapshot();
        assert_eq!(phase(&s, "good").status, PhaseStatus::Done);
        assert_eq!(phase(&s, "bad").status, PhaseStatus::Failed);
        assert_eq!(phase(&s, "bad").detail.as_deref(), Some("boom"));
    }

    #[test]
    fn the_record_is_readable_from_another_thread_while_the_startup_writes_it() {
        let boot = std::sync::Arc::new(BootTracker::new());
        let ids: Vec<String> = (0..50).map(|n| format!("p{n}")).collect();
        let declared: Vec<(&str, &str)> = ids.iter().map(|i| (i.as_str(), "phase")).collect();
        boot.declare(&declared);

        let reader = {
            let boot = boot.clone();
            std::thread::spawn(move || {
                let mut last_done = 0;
                // Reads until the writer says it is over; every snapshot must be coherent.
                loop {
                    let s = boot.snapshot();
                    assert_eq!(s.phases.len(), 50);
                    let done = s
                        .phases
                        .iter()
                        .filter(|p| p.status == PhaseStatus::Done)
                        .count();
                    assert!(done >= last_done, "a phase never goes back");
                    last_done = done;
                    if s.finished {
                        return done;
                    }
                }
            })
        };
        for id in &ids {
            boot.start(id);
            boot.done(id);
        }
        boot.finish();
        assert_eq!(reader.join().unwrap(), 50);
    }

    #[test]
    fn start_server_declares_enough_phases_and_drives_every_one_of_them() {
        // `start_server` needs live databases, so it cannot run here. What can be checked is that
        // the declaration and the code agree: a phase declared in `BOOT_PHASES` but never started
        // and ended in `start_server` would sit at "pending" forever on the splash.
        let phases = crate::BOOT_PHASES;
        assert!(phases.len() >= 8, "only {} phases", phases.len());
        let ids: std::collections::HashSet<_> = phases.iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), phases.len(), "a phase id is declared twice");

        let source = include_str!("lib.rs");
        for (id, label) in phases {
            assert!(!label.trim().is_empty(), "{id} has no label");
            let quoted = format!("\"{id}\"");
            // One occurrence is the declaration; a start and an end make at least two more.
            let uses = source.matches(&quoted).count();
            assert!(
                uses >= 3,
                "phase {id} is declared but start_server drives it {} time(s)",
                uses.saturating_sub(1)
            );
            // And it is started (or progressed) somewhere, not only ended.
            let started = source.contains(&format!("start({quoted})"));
            assert!(started, "phase {id} is never started in start_server");
        }
    }

    #[test]
    fn a_secret_in_an_error_never_reaches_the_record() {
        let boot = BootTracker::new();
        boot.start("db");
        boot.fail(
            "db",
            "could not connect to bolt://neo4j:s3cret@db.internal:7687/db and http://k:topsecret@meili:7700",
        );
        boot.done_with("events", "using nats://user:hunter2@nats:4222");
        let json = serde_json::to_string(&boot.snapshot()).unwrap();
        for secret in ["s3cret", "topsecret", "hunter2"] {
            assert!(!json.contains(secret), "{secret} leaked: {json}");
        }
        assert!(json.contains("***@db.internal:7687"), "{json}");
    }

    #[test]
    fn redact_leaves_ordinary_text_and_urls_without_credentials_alone() {
        for text in [
            "plain message",
            "see https://example.com/path?x=1",
            "bolt://localhost:7687",
            "a@b.c is an address, not a URL authority",
            "",
        ] {
            assert_eq!(redact(text), text);
        }
        assert_eq!(redact("x://u:p@h:1/ y://a@b"), "x://***@h:1/ y://***@b");
    }

    #[test]
    fn the_snapshot_has_the_shape_consumers_rely_on() {
        let boot = BootTracker::new();
        boot.declare(&[("bind", "Reserving the port")]);
        boot.start("bind");
        let value = serde_json::to_value(boot.snapshot()).unwrap();
        assert_eq!(value["finished"], false);
        assert_eq!(value["phases"][0]["id"], "bind");
        assert_eq!(value["phases"][0]["label"], "Reserving the port");
        assert_eq!(
            value["phases"][0]["status"], "running",
            "snake_case, as the splash will read it"
        );
        assert!(value["phases"][0]["done"].is_null());
        assert!(value["started_unix_ms"].as_u64().unwrap() > 1_700_000_000_000);
        assert!(value["version"].is_string());
    }

    #[test]
    fn the_process_wide_tracker_is_one_instance() {
        assert!(std::ptr::eq(tracker(), tracker()));
    }
}
