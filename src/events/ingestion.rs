//! Ingestion activity events — the orchestrator's own work, made observable.
//!
//! The sync pipeline used to be silent: `tracing::info!` lines, and a single
//! `Project/Synced` once everything was over. A client watching the graph could
//! not tell whether the orchestrator was indexing anything at all.
//!
//! This module closes that gap with two ordinary [`CrudEvent`]s carrying
//! `entity_type: "ingestion"`:
//!
//! - [`CrudAction::Progress`] while work happens — phase, counters, current file;
//! - [`CrudAction::Synced`] once, at the end, with the same counters the
//!   `Project/Synced` event carries.
//!
//! No new WebSocket frame is involved: existing clients already read CRUD
//! events, and `?entity_types=ingestion` isolates this stream.
//!
//! ## Why coalescing is not optional
//!
//! A sync of a few thousand files must not produce a few thousand events. The
//! local bus is a `tokio::sync::broadcast`: a slow subscriber that lags has
//! events *dropped*, and CRUD events are forwarded one WebSocket message each
//! (only graph events are batched). So progress is throttled:
//!
//! - **inside a phase**, at most one event per [`INGESTION_PROGRESS_INTERVAL`];
//! - **on a phase change**, always one event — bounded, since a sync has a fixed
//!   number of phases;
//! - **on completion**, always the final event.
//!
//! The result is a stream bounded by `phases + elapsed / interval`, never by the
//! number of files.
//!
//! [`CrudEvent`]: super::types::CrudEvent

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use uuid::Uuid;

use super::types::{EntityType, EventEmitter};

/// Minimum delay between two progress events **inside the same phase**.
///
/// A phase change is never throttled: a sync has a fixed number of phases, so
/// forwarding each transition keeps the stream bounded while letting a client
/// show "reading", "parsing", "storing" as distinct steps.
pub const INGESTION_PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// The step of the ingestion pipeline currently running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestionPhase {
    /// Walking the directory tree.
    Scanning,
    /// Reading file contents from disk.
    Reading,
    /// Parsing with Tree-sitter.
    Parsing,
    /// Writing files, symbols and relations to the graph store.
    Storing,
    /// Indexing code documents in the search engine.
    Indexing,
    /// Computing embeddings for files and functions.
    Embedding,
    /// Removing files that no longer exist.
    Cleanup,
    /// Ingestion finished (only carried by the final event).
    Done,
}

impl IngestionPhase {
    /// The wire value, matching the `snake_case` serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            IngestionPhase::Scanning => "scanning",
            IngestionPhase::Reading => "reading",
            IngestionPhase::Parsing => "parsing",
            IngestionPhase::Storing => "storing",
            IngestionPhase::Indexing => "indexing",
            IngestionPhase::Embedding => "embedding",
            IngestionPhase::Cleanup => "cleanup",
            IngestionPhase::Done => "done",
        }
    }
}

/// Counters carried by every progress event.
#[derive(Debug, Clone, Copy, Default)]
pub struct IngestionCounters {
    /// Files discovered by the scan (0 until the scan is over).
    pub files_total: usize,
    /// Files already handled by the current phase.
    pub files_done: usize,
    /// Files skipped because their hash did not change.
    pub files_skipped: usize,
    /// Files removed from the graph as stale.
    pub files_deleted: usize,
}

/// Payload of an `ingestion/progress` event.
#[derive(Debug, Clone, Serialize)]
pub struct IngestionProgress {
    /// Human-readable project identifier, for display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_slug: Option<String>,
    /// Current phase.
    pub phase: IngestionPhase,
    /// Files discovered by the scan.
    pub files_total: usize,
    /// Files handled so far.
    pub files_done: usize,
    /// Files skipped (unchanged).
    pub files_skipped: usize,
    /// Files removed as stale.
    pub files_deleted: usize,
    /// One file being handled right now — a sample, not an exhaustive list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_path: Option<String>,
    /// Milliseconds since the sync started.
    pub elapsed_ms: u64,
}

/// Final counters of a sync, mirroring the `Project/Synced` payload.
#[derive(Debug, Clone, Copy, Default)]
pub struct IngestionSummary {
    pub files_parsed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub errors: usize,
    pub duration_ms: u64,
    /// `None` when the sync ran outside a project (legacy directory sync) or
    /// when the project could not be read: the field is then omitted rather
    /// than guessed.
    pub is_first_sync: Option<bool>,
}

/// Payload of an `ingestion/synced` event.
///
/// Deliberately mirrors the `Project/Synced` payload (`files_parsed`,
/// `files_skipped`, `files_deleted`, `errors`, `duration_ms`, `is_first_sync`)
/// so a client reads the same counts whichever event it happens to catch, plus
/// `phase: "done"` so the ingestion stream is self-terminating.
#[derive(Debug, Clone, Serialize)]
pub struct IngestionSynced {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_slug: Option<String>,
    pub phase: IngestionPhase,
    pub files_parsed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub errors: usize,
    pub duration_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_first_sync: Option<bool>,
}

/// Emits coalesced ingestion activity for one sync run.
///
/// Cheap to construct and safe to hold across awaits inside `sync_*`: it owns an
/// `Arc` to the shared emitter and two `Instant`s. With no emitter (the
/// orchestrator can be built without one, e.g. in tests and CLI paths) every
/// method is a no-op.
pub struct IngestionReporter {
    emitter: Option<Arc<dyn EventEmitter>>,
    project_id: Option<Uuid>,
    project_slug: Option<String>,
    started: Instant,
    last_emit: Option<Instant>,
    phase: IngestionPhase,
    interval: Duration,
    emitted: usize,
}

impl IngestionReporter {
    /// Create a reporter for one sync run of one project.
    pub fn new(
        emitter: Option<Arc<dyn EventEmitter>>,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
    ) -> Self {
        Self::with_interval(
            emitter,
            project_id,
            project_slug,
            INGESTION_PROGRESS_INTERVAL,
        )
    }

    /// Same as [`Self::new`] with an explicit throttle — the only reason to use
    /// it is a test that cannot wait 250 ms between assertions.
    pub fn with_interval(
        emitter: Option<Arc<dyn EventEmitter>>,
        project_id: Option<Uuid>,
        project_slug: Option<&str>,
        interval: Duration,
    ) -> Self {
        Self {
            emitter,
            project_id,
            project_slug: project_slug.map(|s| s.to_string()),
            started: Instant::now(),
            last_emit: None,
            phase: IngestionPhase::Scanning,
            interval,
            emitted: 0,
        }
    }

    /// How many events this run has emitted — used by tests and by the final
    /// `tracing` line.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    /// Enter a phase. Always emitted: a sync has a bounded number of phases.
    pub fn enter_phase(&mut self, phase: IngestionPhase, counters: IngestionCounters) {
        self.enter_phase_at(phase, counters, None);
    }

    /// Same as [`Self::enter_phase`], carrying the file being handled.
    ///
    /// Needed for the single-file sync path, where the only event of the run is
    /// the phase transition itself — a follow-up `tick` would be throttled away.
    pub fn enter_phase_at(
        &mut self,
        phase: IngestionPhase,
        counters: IngestionCounters,
        current_path: Option<&str>,
    ) {
        self.phase = phase;
        self.emit(counters, current_path, true);
    }

    /// Report fine-grained progress (typically one call per file). Throttled.
    pub fn tick(&mut self, counters: IngestionCounters, current_path: Option<&str>) {
        self.emit(counters, current_path, false);
    }

    /// Emit the terminal event. Always emitted, even with no emitter configured
    /// the call is simply ignored.
    pub fn synced(&mut self, summary: IngestionSummary) {
        self.phase = IngestionPhase::Done;
        let Some(emitter) = self.emitter.clone() else {
            return;
        };
        let payload = IngestionSynced {
            project_slug: self.project_slug.clone(),
            phase: IngestionPhase::Done,
            files_parsed: summary.files_parsed,
            files_skipped: summary.files_skipped,
            files_deleted: summary.files_deleted,
            errors: summary.errors,
            duration_ms: summary.duration_ms,
            is_first_sync: summary.is_first_sync,
        };
        let payload = match serde_json::to_value(&payload) {
            Ok(value) => value,
            Err(_) => return,
        };
        emitter.emit_synced(
            EntityType::Ingestion,
            &self.entity_id(),
            payload,
            self.project_id.map(|p| p.to_string()),
        );
        self.emitted += 1;
    }

    fn entity_id(&self) -> String {
        self.project_id
            .map(|p| p.to_string())
            .unwrap_or_else(|| "directory".to_string())
    }

    fn emit(&mut self, counters: IngestionCounters, current_path: Option<&str>, force: bool) {
        if self.emitter.is_none() {
            return;
        }
        let now = Instant::now();
        if !force {
            match self.last_emit {
                Some(last) if now.duration_since(last) < self.interval => return,
                _ => {}
            }
        }
        let progress = IngestionProgress {
            project_slug: self.project_slug.clone(),
            phase: self.phase,
            files_total: counters.files_total,
            files_done: counters.files_done,
            files_skipped: counters.files_skipped,
            files_deleted: counters.files_deleted,
            current_path: current_path.map(|p| p.to_string()),
            elapsed_ms: now.duration_since(self.started).as_millis() as u64,
        };
        let payload = match serde_json::to_value(&progress) {
            Ok(value) => value,
            Err(_) => return,
        };
        if let Some(emitter) = self.emitter.clone() {
            emitter.emit_progress(
                EntityType::Ingestion,
                &self.entity_id(),
                payload,
                self.project_id.map(|p| p.to_string()),
            );
            self.last_emit = Some(now);
            self.emitted += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CrudAction, CrudEvent, EventBus, HybridEmitter};

    fn reporter(
        interval_ms: u64,
    ) -> (
        IngestionReporter,
        tokio::sync::broadcast::Receiver<CrudEvent>,
    ) {
        let hybrid = Arc::new(HybridEmitter::new(Arc::new(EventBus::default())));
        let rx = hybrid.subscribe();
        let emitter: Arc<dyn EventEmitter> = hybrid;
        let reporter = IngestionReporter::with_interval(
            Some(emitter),
            Some(Uuid::new_v4()),
            Some("demo"),
            Duration::from_millis(interval_ms),
        );
        (reporter, rx)
    }

    fn drain(rx: &mut tokio::sync::broadcast::Receiver<CrudEvent>) -> Vec<CrudEvent> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            out.push(event);
        }
        out
    }

    #[test]
    fn phase_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(IngestionPhase::Scanning).unwrap(),
            serde_json::json!("scanning")
        );
        assert_eq!(IngestionPhase::Done.as_str(), "done");
    }

    #[test]
    fn progress_event_carries_the_contract_shape() {
        let (mut reporter, mut rx) = reporter(0);
        reporter.enter_phase(
            IngestionPhase::Parsing,
            IngestionCounters {
                files_total: 12,
                files_done: 3,
                ..Default::default()
            },
        );
        let events = drain(&mut rx);
        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(event.entity_type, EntityType::Ingestion);
        assert_eq!(event.action, CrudAction::Progress);
        assert_eq!(event.payload["phase"], "parsing");
        assert_eq!(event.payload["files_total"], 12);
        assert_eq!(event.payload["files_done"], 3);
        assert_eq!(event.payload["project_slug"], "demo");
        // Absent, not null — the contract says so.
        assert!(event.payload.get("current_path").is_none());
    }

    #[test]
    fn per_file_ticks_are_throttled_but_phase_changes_are_not() {
        // A long interval makes "throttled" observable without sleeping.
        let (mut reporter, mut rx) = reporter(10_000);
        for i in 0..100 {
            reporter.tick(
                IngestionCounters {
                    files_total: 100,
                    files_done: i,
                    ..Default::default()
                },
                Some("src/a.rs"),
            );
        }
        // First tick always goes through, the remaining 99 are throttled.
        assert_eq!(drain(&mut rx).len(), 1);

        reporter.enter_phase(IngestionPhase::Storing, IngestionCounters::default());
        let events = drain(&mut rx);
        assert_eq!(events.len(), 1, "a phase change is never throttled");
        assert_eq!(events[0].payload["phase"], "storing");
    }

    #[test]
    fn synced_is_always_emitted_and_mirrors_project_synced() {
        let (mut reporter, mut rx) = reporter(10_000);
        reporter.tick(IngestionCounters::default(), None);
        reporter.synced(IngestionSummary {
            files_parsed: 42,
            files_skipped: 7,
            files_deleted: 2,
            errors: 0,
            duration_ms: 1234,
            is_first_sync: Some(true),
        });
        let events = drain(&mut rx);
        assert_eq!(events.len(), 2);
        let last = &events[1];
        assert_eq!(last.action, CrudAction::Synced);
        assert_eq!(last.payload["phase"], "done");
        assert_eq!(last.payload["files_parsed"], 42);
        assert_eq!(last.payload["files_skipped"], 7);
        assert_eq!(last.payload["files_deleted"], 2);
        assert_eq!(last.payload["errors"], 0);
        assert_eq!(last.payload["duration_ms"], 1234);
        assert_eq!(last.payload["is_first_sync"], true);
    }

    #[test]
    fn a_reporter_without_emitter_is_a_no_op() {
        let mut reporter = IngestionReporter::new(None, None, None);
        reporter.enter_phase(IngestionPhase::Scanning, IngestionCounters::default());
        reporter.tick(IngestionCounters::default(), Some("x"));
        reporter.synced(IngestionSummary::default());
        assert_eq!(reporter.emitted(), 0);
    }
}
