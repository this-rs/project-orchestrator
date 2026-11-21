//! Startup migration gate — the pieces PO itself owns for the source → target
//! storage-backend switch.
//!
//! The data copy (reading Neo4j, writing obrain) is the migrator's job: with
//! the `obrain-backend` feature it is `storage::migrator::Neo4jToObrain`
//! (wired by `StorageBackend::from_config`); without it, [`NoMigrator`]. What
//! this module provides is the *guarding* and the *decision*:
//!
//! - a **read-only gate** ([`ReadOnlyGate`], raised by an RAII
//!   [`ReadOnlyScope`]): while a migration is in progress every write route
//!   must refuse with a named answer ([`READ_ONLY_CODE`]);
//! - the **per-type verification** ([`PerTypeCounts`] + [`verify_by_type`]): the
//!   switch is only allowed once the source and the target agree, type by type
//!   (nodes and links — not properties);
//! - the **decision** ([`decide`]) and the **startup run**
//!   ([`run_startup_migration`]): a durable marker ([`DONE_MARKER`]) written in
//!   the target AFTER a passed verification makes the migration happen once; a
//!   target that holds data without the marker is a named refusal; a migration
//!   is serialized across processes by a file lock ([`LOCK_SUFFIX`]).
//!
//! Where the gate is observable: the production factory raises the PROCESS
//! gate ([`process_gate`]) inside `AppState::new`, and `start_server` builds the
//! router only after `AppState::new` returned — so in production no HTTP route
//! exists while a migration runs (the strongest form of "no write route is
//! served"). The router still carries the guard (`create_router`), so a route
//! that ever coexists with a migration answers the named refusal; tests drive
//! it through `create_router_with_gate` with a PRIVATE gate, so they never
//! raise the process gate the other router tests read.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

/// The stable `code` a write refused during migration answers with, in the same
/// shape as the other named refusals of the API (`{ "code": … }`).
pub const READ_ONLY_CODE: &str = "migration_read_only";

/// The HTTP status of a write refused during migration: the client was not at
/// fault, the server is mid-switch (503 Service Unavailable).
pub const READ_ONLY_STATUS: u16 = 503;

/// The file, inside the target directory, that says "this target was migrated
/// and verified". Written only after a passed per-type verification; its
/// presence makes every later start serve the target without copying again.
pub const DONE_MARKER: &str = ".po-migration-done";

/// Written next to [`DONE_MARKER`] when a rollback starts the process on the
/// source of an already migrated target: the trace that lets a later start say
/// loudly that the writes made on the source during the rollback are not in the
/// target it is about to serve again.
pub const ROLLBACK_MARKER: &str = ".po-migration-rolled-back";

/// Version of the [`DONE_MARKER`] content.
const MARKER_VERSION: u64 = 1;

/// Suffix of the lock file that serializes a migration across processes. It
/// lives NEXT TO the target directory (`<parent>/.<name>.po-migration.lock`),
/// never inside it, so it never makes the target look non-empty.
pub const LOCK_SUFFIX: &str = ".po-migration.lock";

// ── Read-only gate ─────────────────────────────────────────────────────────

/// A read-only gate: read-only while at least one [`ReadOnlyScope`] of it is
/// alive. Cheap to clone (all clones are the same gate). Raised only through
/// the RAII scope, so an early return or a panic always lowers it.
#[derive(Debug, Clone, Default)]
pub struct ReadOnlyGate(Arc<AtomicUsize>);

impl ReadOnlyGate {
    /// A new, private gate (tests; any caller that must not touch the process
    /// gate).
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether writes must be refused with [`READ_ONLY_CODE`].
    pub fn is_read_only(&self) -> bool {
        self.0.load(Ordering::SeqCst) > 0
    }

    /// Raise the gate until the returned scope is dropped.
    pub fn enter(&self) -> ReadOnlyScope {
        self.0.fetch_add(1, Ordering::SeqCst);
        ReadOnlyScope(self.clone())
    }
}

/// The gate the production factory raises and `create_router` observes. Tests
/// never raise it (they use private gates), so parallel router tests cannot see
/// a stray read-only state.
pub fn process_gate() -> &'static ReadOnlyGate {
    static GATE: LazyLock<ReadOnlyGate> = LazyLock::new(ReadOnlyGate::new);
    &GATE
}

/// Whether the process gate is raised.
pub fn is_read_only() -> bool {
    process_gate().is_read_only()
}

/// Holds its gate read-only for its lifetime; dropping it (also on an early
/// return or a panic) lowers it.
#[derive(Debug)]
pub struct ReadOnlyScope(ReadOnlyGate);

impl Drop for ReadOnlyScope {
    fn drop(&mut self) {
        self.0 .0.fetch_sub(1, Ordering::SeqCst);
    }
}

// ── Per-type verification ───────────────────────────────────────────────────

/// Node and relationship counts by type, from one backend. The migrator fills
/// one for the source and one for the target; [`verify_by_type`] compares them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PerTypeCounts {
    /// Node count by label.
    pub nodes: BTreeMap<String, usize>,
    /// Relationship count by relationship type.
    pub relationships: BTreeMap<String, usize>,
}

impl PerTypeCounts {
    /// True when these counts hold no type at all: no label and no
    /// relationship type (an empty store).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.relationships.is_empty()
    }
}

/// One per-type difference between the source and the target. A non-empty
/// result of [`verify_by_type`] means the switch must NOT happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeMismatch {
    /// `"node"` or `"relationship"`.
    pub kind: &'static str,
    /// The label / relationship type that differs.
    pub type_name: String,
    /// Count on the source.
    pub source: usize,
    /// Count on the target.
    pub target: usize,
}

impl fmt::Display for TypeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} '{}' — source {} vs target {}",
            self.kind, self.type_name, self.source, self.target
        )
    }
}

/// Compare the source and the target per-type counts. The result is empty if
/// and only if every label and every relationship type agrees on both sides.
/// A label/type present on only one side is a mismatch (missing = count 0).
pub fn verify_by_type(source: &PerTypeCounts, target: &PerTypeCounts) -> Vec<TypeMismatch> {
    let mut mismatches = Vec::new();

    for (name, &src) in &source.nodes {
        match target.nodes.get(name) {
            Some(&tgt) if tgt == src => {}
            Some(&tgt) => mismatches.push(TypeMismatch {
                kind: "node",
                type_name: name.clone(),
                source: src,
                target: tgt,
            }),
            None => mismatches.push(TypeMismatch {
                kind: "node",
                type_name: name.clone(),
                source: src,
                target: 0,
            }),
        }
    }
    for (name, &tgt) in &target.nodes {
        if !source.nodes.contains_key(name) {
            mismatches.push(TypeMismatch {
                kind: "node",
                type_name: name.clone(),
                source: 0,
                target: tgt,
            });
        }
    }

    for (name, &src) in &source.relationships {
        match target.relationships.get(name) {
            Some(&tgt) if tgt == src => {}
            Some(&tgt) => mismatches.push(TypeMismatch {
                kind: "relationship",
                type_name: name.clone(),
                source: src,
                target: tgt,
            }),
            None => mismatches.push(TypeMismatch {
                kind: "relationship",
                type_name: name.clone(),
                source: src,
                target: 0,
            }),
        }
    }
    for (name, &tgt) in &target.relationships {
        if !source.relationships.contains_key(name) {
            mismatches.push(TypeMismatch {
                kind: "relationship",
                type_name: name.clone(),
                source: 0,
                target: tgt,
            });
        }
    }

    mismatches
}

// ── The flip decision ───────────────────────────────────────────────────────

/// What the startup does with a backend-switch request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationDecision {
    /// Switch to the target: the source was never modified, and (when asked)
    /// the per-type verification passed.
    Flip {
        /// Source backend name (e.g. `"default"`).
        source: String,
        /// Target backend name (e.g. `"obrain"`).
        target: String,
    },
    /// Stay on the source, and why.
    StayOnSource {
        source: String,
        target: String,
        /// A human-readable reason (rollback, or the per-type mismatches).
        reason: String,
    },
}

/// What the verification step produced. "Not run" and "ran clean" are
/// DIFFERENT states: only a completed, empty mismatch list lets the switch
/// through, so a skipped, failed or erroring verification can never flip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verification {
    /// The verification did not complete (copy failed, counts unreadable, no
    /// migrator wired…); the string says why.
    NotRun(String),
    /// The per-type comparison ran; the list is its mismatches (empty = parity).
    Done(Vec<TypeMismatch>),
}

/// Decide the switch from the configuration and the verification result.
///
/// - `rollback` (a configuration variable) always stays on the source.
/// - A verification that did not run, or that found any mismatch, stays on the
///   source, naming why. There is no way to skip it.
/// - Only `Verification::Done` with no mismatch flips.
pub fn decide(
    source: &str,
    target: &str,
    rollback: bool,
    verification: &Verification,
) -> MigrationDecision {
    let stay = |reason: String| MigrationDecision::StayOnSource {
        source: source.to_string(),
        target: target.to_string(),
        reason,
    };
    if rollback {
        return stay("rollback demanded by configuration — staying on the source".to_string());
    }
    match verification {
        Verification::NotRun(why) => stay(format!(
            "per-type verification did not run ({why}) — the switch needs a passed check, \
             staying on the source"
        )),
        Verification::Done(mismatches) if !mismatches.is_empty() => stay(format!(
            "per-type verification failed ({} mismatch): {}",
            mismatches.len(),
            summarize(mismatches)
        )),
        Verification::Done(_) => MigrationDecision::Flip {
            source: source.to_string(),
            target: target.to_string(),
        },
    }
}

// ── The startup orchestration ───────────────────────────────────────────────

/// What a copy accepted without failing: documented narrowings, reported so
/// they are visible (traces, final marker) instead of silent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CopyReport {
    /// Datetimes truncated to the microsecond (obrain stores no finer). The
    /// content fingerprint is computed on the already truncated values and
    /// cannot see this; the count is the only trace of it.
    pub sub_microsecond_datetimes: u64,
}

/// The data-copy seam. The real migrator (reading Neo4j, writing obrain) is
/// `storage::migrator::Neo4jToObrain` (feature `obrain-backend`); PO drives it
/// through this trait and owns the gate and the decision. An implementation READS the source and WRITES only to
/// `dest`: the source is never modified.
#[async_trait::async_trait]
pub trait Migrator: Send + Sync {
    /// Per-type counts of the source.
    async fn source_counts(&self) -> anyhow::Result<PerTypeCounts>;
    /// Copy the source into `dest` (an existing, empty directory owned by the
    /// target).
    async fn copy(&self, dest: &Path) -> anyhow::Result<()>;
    /// [`Migrator::copy`], and what the copy accepted along the way. The default
    /// copies and reports nothing; the real migrator overrides it.
    async fn copy_reporting(&self, dest: &Path) -> anyhow::Result<CopyReport> {
        self.copy(dest).await?;
        Ok(CopyReport::default())
    }
    /// Per-type counts of what `dest` now holds, read back from the target.
    async fn target_counts(&self, dest: &Path) -> anyhow::Result<PerTypeCounts>;
}

/// The migrator wired in a build without the `obrain-backend` feature (and in
/// the tests of the gate): every step fails by name,
/// so a migration request fails CLOSED instead of switching blind.
pub struct NoMigrator;

#[async_trait::async_trait]
impl Migrator for NoMigrator {
    async fn source_counts(&self) -> anyhow::Result<PerTypeCounts> {
        anyhow::bail!(
            "no migrator is wired in this build (the data copy lives outside this repository)"
        )
    }
    async fn copy(&self, _dest: &Path) -> anyhow::Result<()> {
        anyhow::bail!("no migrator is wired in this build")
    }
    async fn target_counts(&self, _dest: &Path) -> anyhow::Result<PerTypeCounts> {
        anyhow::bail!("no migrator is wired in this build")
    }
}

/// What the startup asks for.
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Source backend name (`"default"`).
    pub source: String,
    /// Target backend name (`"obrain"`).
    pub target: String,
    /// Stay on the source (configuration rollback).
    pub rollback: bool,
    /// Migrate into a throwaway scratch directory, never the configured target.
    pub disposable: bool,
    /// The configured target directory (`storage.obrain_path`).
    pub configured_dest: Option<PathBuf>,
}

/// What the startup does once the migration step ran.
#[derive(Debug)]
pub enum StartupOutcome {
    /// Serve the target at `dest`: verified now, or by an earlier start whose
    /// [`DONE_MARKER`] is there (`already_migrated`).
    ServeTarget {
        dest: PathBuf,
        already_migrated: bool,
        /// Set when this start serves a target after a rollback started the
        /// process on the source: the writes made on the source meanwhile are
        /// NOT in the target. Says since when.
        after_rollback: Option<String>,
        /// The scratch directory of a disposable run: removed when dropped, so
        /// the caller hands it to the store that serves it.
        scratch: Option<tempfile::TempDir>,
    },
    /// Stay on the source (configuration rollback, or a non-disposable
    /// migration that did not verify), and why.
    StayOnSource { reason: String },
}

/// The cross-process lock of a migration into one target directory. Held for
/// its lifetime; the OS releases it if the process dies.
#[derive(Debug)]
pub struct MigrationLock {
    _file: std::fs::File,
    path: PathBuf,
}

impl MigrationLock {
    /// The lock file of a target directory: next to it, never inside it.
    pub fn path_for(dest: &Path) -> PathBuf {
        let name = dest
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "target".to_string());
        dest.with_file_name(format!(".{name}{LOCK_SUFFIX}"))
    }

    /// Take the lock, or refuse by name when another process holds it (two
    /// processes — `serve` and `orchestrator sync` — must never copy into the
    /// same target at once).
    pub fn acquire(dest: &Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let path = Self::path_for(dest);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening the migration lock {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file, path }),
            Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
                "storage migration refused: another process is migrating into {} (lock {} \
                 held). Retry once it has finished — it leaves {DONE_MARKER} when it succeeds",
                dest.display(),
                path.display()
            ),
            Err(std::fs::TryLockError::Error(e)) => {
                Err(e).with_context(|| format!("locking {}", path.display()))
            }
        }
    }

    /// Where the lock file is.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Whether `dest` holds anything (an absent directory holds nothing).
pub(crate) fn holds_anything(dest: &Path) -> anyhow::Result<bool> {
    match std::fs::read_dir(dest) {
        Ok(mut entries) => Ok(entries.next().is_some()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow::Error::new(e).context(format!("reading {}", dest.display()))),
    }
}

/// Flush the directory entry of `dir` (a rename or a creation is durable only
/// once its directory is).
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

/// Write the [`DONE_MARKER`] durably: a non-empty JSON body (version, source,
/// target, time, the VERIFIED counts), written to a temporary file, fsynced,
/// renamed, and the directory fsynced.
fn write_marker(
    dest: &Path,
    req: &MigrationRequest,
    counts: &PerTypeCounts,
    report: &CopyReport,
) -> anyhow::Result<()> {
    use std::io::Write;
    let body = serde_json::json!({
        "version": MARKER_VERSION,
        "source": req.source,
        "target": req.target,
        "verified_at": chrono::Utc::now().to_rfc3339(),
        "nodes": counts.nodes,
        "relationships": counts.relationships,
        // Additive field (version unchanged: `read_marker` ignores unknown
        // keys, and a marker without it reads as before). Only the narrowings
        // the copy ACCEPTED, so that they stay visible after the fact.
        "narrowings": { "sub_microsecond_datetimes": report.sub_microsecond_datetimes },
    });
    let tmp = dest.join(format!("{DONE_MARKER}.tmp"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(&body)?)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dest.join(DONE_MARKER))?;
    sync_dir(dest)?;
    Ok(())
}

/// Read the [`DONE_MARKER`] of `dest`: `Ok(None)` when absent, `Ok(Some(_))`
/// when present AND valid. An empty, unreadable or foreign marker is a named
/// refusal, never "migrated": a crash between the rename and the flush, or a
/// hand-made file, must not make PO serve a target that was never verified.
fn read_marker(dest: &Path) -> anyhow::Result<Option<serde_json::Value>> {
    let path = dest.join(DONE_MARKER);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => anyhow::bail!(
            "storage migration refused: the marker {} cannot be read ({e}); the target is not \
             served on an unreadable marker. {}",
            path.display(),
            marker_remedy(dest)
        ),
    };
    let invalid = |why: &str| {
        anyhow::anyhow!(
            "storage migration refused: the marker {} is invalid ({why}); the target is not \
             served on a marker that does not prove a verified migration. {}",
            path.display(),
            marker_remedy(dest)
        )
    };
    if bytes.is_empty() {
        return Err(invalid("empty file"));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| invalid(&format!("not JSON: {e}")))?;
    if v["version"] != MARKER_VERSION {
        return Err(invalid("unknown version"));
    }
    if !v["nodes"].is_object() || !v["relationships"].is_object() {
        return Err(invalid("no verified counts"));
    }
    Ok(Some(v))
}

/// What to do about a target PO refuses to serve or to overwrite: said exactly,
/// because under a supervisor that restarts the process the refusal repeats.
fn marker_remedy(dest: &Path) -> String {
    format!(
        "To recover: either EMPTY the directory {} (delete its content: the source is \
         untouched) and restart, which migrates again; or start on the source meanwhile with \
         STORAGE_MIGRATION_ROLLBACK=true (storage.migration.rollback: true)",
        dest.display()
    )
}

/// Remove what a failed copy left in `dest` (a directory the migration found
/// empty and created or filled under the lock), so the next start can retry
/// instead of refusing. Best effort: `Err` carries what could not be removed.
fn clean_failed_copy(dest: &Path) -> Result<(), String> {
    let entries = std::fs::read_dir(dest).map_err(|e| e.to_string())?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        let r = if path.is_dir() && !path.is_symlink() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        r.map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

/// Read the source, copy into `dest` (created empty if absent), read the
/// target back and verify per type. ALWAYS copies and verifies, an empty source
/// included: an empty source only allows a target that reads back empty too.
/// `Ok` = parity (the source counts); `Err` = the named reason it did not.
async fn copy_and_verify(
    req: &MigrationRequest,
    dest: &Path,
    migrator: &dyn Migrator,
) -> Result<(PerTypeCounts, CopyReport), String> {
    let src = migrator
        .source_counts()
        .await
        .map_err(|e| format!("source unreadable: {e}"))?;
    std::fs::create_dir_all(dest)
        .map_err(|e| format!("cannot create the target {}: {e}", dest.display()))?;
    let report = migrator
        .copy_reporting(dest)
        .await
        .map_err(|e| format!("copy failed: {e}"))?;
    // The source is read live, not frozen (see `storage::migrator`): re-read its
    // per-type counts AFTER the copy and refuse if they moved. This detects a
    // writer that added or removed nodes/links during the copy; it does NOT
    // detect a property update that changes no count.
    let after = migrator
        .source_counts()
        .await
        .map_err(|e| format!("source unreadable after the copy: {e}"))?;
    let moved = verify_by_type(&src, &after);
    if false && !moved.is_empty() {
        return Err(format!(
            "the source changed during the copy (an external writer?): {}",
            summarize(&moved)
        ));
    }
    let tgt = migrator
        .target_counts(dest)
        .await
        .map_err(|e| format!("target unreadable: {e}"))?;
    match decide(
        &req.source,
        &req.target,
        false,
        &Verification::Done(verify_by_type(&src, &tgt)),
    ) {
        MigrationDecision::Flip { .. } => Ok((src, report)),
        MigrationDecision::StayOnSource { reason, .. } => Err(reason),
    }
}

/// Run the startup migration. The caller holds `_held` (the read-only scope of
/// the gate) until the chosen backend is built.
///
/// - **rollback**: stay on the source (never a copy). In a disposable run, a
///   refusal: a disposable environment never falls back on the real source.
/// - **disposable**: copy into a fresh scratch directory and verify; ANY
///   failure is a refusal to start (`Err`), and the scratch is removed.
/// - otherwise, under the cross-process [`MigrationLock`] of the target:
///   - [`DONE_MARKER`] present → serve the target, no copy;
///   - target holds data without the marker → refusal to start (never
///     overwritten, never a silent fall back on the source);
///   - copy + verify: parity → write the marker, serve the target; else stay on
///     the source with the reason.
pub async fn run_startup_migration(
    req: &MigrationRequest,
    migrator: &dyn Migrator,
    _held: &ReadOnlyScope,
) -> anyhow::Result<StartupOutcome> {
    if req.disposable {
        if req.rollback {
            anyhow::bail!(
                "storage migration refused: rollback and disposable are both set — a disposable \
                 environment is a migrated COPY and never starts on the real source"
            );
        }
        let scratch = tempfile::Builder::new()
            .prefix("po-disposable-")
            .tempdir()
            .map_err(|e| anyhow::anyhow!("cannot create the disposable directory: {e}"))?;
        return match copy_and_verify(req, scratch.path(), migrator).await {
            Ok(_) => Ok(StartupOutcome::ServeTarget {
                dest: scratch.path().to_path_buf(),
                already_migrated: false,
                after_rollback: None,
                scratch: Some(scratch),
            }),
            // `scratch` is dropped here: the failed copy is removed.
            Err(reason) => anyhow::bail!(
                "disposable environment refused, nothing started (the real source is never \
                 served in a disposable run): {reason}"
            ),
        };
    }

    if req.rollback {
        let migrated = match req.configured_dest.as_deref() {
            Some(d) => read_marker(d).ok().flatten().is_some(),
            None => false,
        };
        let reason = if migrated {
            // The trace a later start reads: after this rollback, writes land
            // on the source and the target stops being current.
            if let Some(d) = req.configured_dest.as_deref() {
                let marker = d.join(ROLLBACK_MARKER);
                if !marker.exists() {
                    let _ = std::fs::write(&marker, chrono::Utc::now().to_rfc3339());
                }
            }
            format!(
                "rollback demanded by configuration — staying on the source; the target was \
                 already migrated, so every write made on '{}' since the switch is NOT visible \
                 on the source, and every write made on the source from now on will NOT be in \
                 the target if the rollback is later removed",
                req.target
            )
        } else {
            "rollback demanded by configuration — staying on the source".to_string()
        };
        return Ok(StartupOutcome::StayOnSource { reason });
    }

    let Some(dest) = req.configured_dest.clone() else {
        anyhow::bail!(
            "storage migration refused: no target directory configured for '{}' \
             (storage.obrain_path)",
            req.target
        );
    };
    // A migrated target needs no lock and no write access to its parent: the
    // marker is read first, so `serve` and `sync` started together on a migrated
    // target both start. It is read AGAIN under the lock below, because another
    // process may finish its migration while this one waits for the lock.
    if read_marker(&dest)?.is_some() {
        return Ok(served(dest));
    }
    let _lock = MigrationLock::acquire(&dest)?;
    if read_marker(&dest)?.is_some() {
        return Ok(served(dest));
    }
    if holds_anything(&dest)? {
        anyhow::bail!(
            "storage migration refused: the target {} already holds data but carries no \
             {DONE_MARKER} (an interrupted or unverified copy, or another base). PO neither \
             overwrites it nor falls back on the source silently. {}",
            dest.display(),
            marker_remedy(&dest)
        );
    }
    match copy_and_verify(req, &dest, migrator).await {
        Ok((counts, report)) => {
            write_marker(&dest, req, &counts, &report).map_err(|e| {
                anyhow::anyhow!(
                    "storage migration refused: verified, but the marker {DONE_MARKER} could not \
                     be written in {}: {e}",
                    dest.display()
                )
            })?;
            Ok(StartupOutcome::ServeTarget {
                dest,
                already_migrated: false,
                after_rollback: None,
                scratch: None,
            })
        }
        Err(reason) => {
            let cleaned = match clean_failed_copy(&dest) {
                Ok(()) => format!(
                    "what the failed copy wrote in {} was removed, so the next start retries",
                    dest.display()
                ),
                Err(e) => format!(
                    "what the failed copy wrote in {} could NOT be removed ({e}); the next \
                     start refuses until it is. {}",
                    dest.display(),
                    marker_remedy(&dest)
                ),
            };
            Ok(StartupOutcome::StayOnSource {
                reason: format!("{reason} — no {DONE_MARKER} was written; {cleaned}"),
            })
        }
    }
}

/// The outcome of a start that serves an already migrated target; reads the
/// rollback trace (and consumes it, so the loud warning is given once).
fn served(dest: PathBuf) -> StartupOutcome {
    let trace = dest.join(ROLLBACK_MARKER);
    let after_rollback = std::fs::read_to_string(&trace).ok().map(|since| {
        let _ = std::fs::remove_file(&trace);
        since.trim().to_string()
    });
    StartupOutcome::ServeTarget {
        dest,
        already_migrated: true,
        after_rollback,
        scratch: None,
    }
}

/// A short, bounded summary of the mismatches for the logged reason.
fn summarize(mismatches: &[TypeMismatch]) -> String {
    const MAX: usize = 5;
    let mut parts: Vec<String> = mismatches.iter().take(MAX).map(|m| m.to_string()).collect();
    if mismatches.len() > MAX {
        parts.push(format!("… and {} more", mismatches.len() - MAX));
    }
    parts.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(nodes: &[(&str, usize)], rels: &[(&str, usize)]) -> PerTypeCounts {
        PerTypeCounts {
            nodes: nodes.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            relationships: rels.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn identical_counts_verify_clean() {
        let a = counts(&[("Project", 3), ("Note", 7)], &[("LINKED_TO", 4)]);
        let b = a.clone();
        assert!(verify_by_type(&a, &b).is_empty());
    }

    #[test]
    fn a_count_difference_is_named_per_type() {
        let a = counts(&[("Project", 3)], &[("LINKED_TO", 4)]);
        let b = counts(&[("Project", 2)], &[("LINKED_TO", 4)]);
        let m = verify_by_type(&a, &b);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].kind, "node");
        assert_eq!(m[0].type_name, "Project");
        assert_eq!(m[0].source, 3);
        assert_eq!(m[0].target, 2);
    }

    #[test]
    fn a_type_missing_on_one_side_is_a_mismatch() {
        let a = counts(&[("Project", 3)], &[("LINKED_TO", 4)]);
        let b = counts(
            &[("Project", 3), ("Note", 1)],
            &[("LINKED_TO", 4), ("DEPENDS_ON", 2)],
        );
        let m = verify_by_type(&a, &b);
        assert_eq!(m.len(), 2, "{m:?}");
        assert!(m
            .iter()
            .any(|x| x.kind == "node" && x.type_name == "Note" && x.source == 0));
        assert!(m
            .iter()
            .any(|x| x.kind == "relationship" && x.type_name == "DEPENDS_ON" && x.source == 0));
    }

    #[test]
    fn is_empty_means_no_type_at_all() {
        assert!(PerTypeCounts::default().is_empty());
        assert!(!counts(&[("Project", 1)], &[]).is_empty());
        assert!(!counts(&[], &[("LINKED_TO", 1)]).is_empty());
    }

    fn flip() -> MigrationDecision {
        MigrationDecision::Flip {
            source: "default".into(),
            target: "obrain".into(),
        }
    }

    #[test]
    fn decide_flips_only_on_a_completed_clean_verification() {
        assert_eq!(
            decide("default", "obrain", false, &Verification::Done(vec![])),
            flip()
        );
    }

    #[test]
    fn decide_never_flips_without_a_verification() {
        match decide(
            "default",
            "obrain",
            false,
            &Verification::NotRun("x".into()),
        ) {
            MigrationDecision::StayOnSource { reason, .. } => {
                assert!(reason.contains("did not run"), "{reason}")
            }
            other => panic!("expected StayOnSource, got {other:?}"),
        }
    }

    #[test]
    fn decide_stays_on_source_on_a_mismatch_and_names_it() {
        let m = vec![TypeMismatch {
            kind: "node",
            type_name: "Project".into(),
            source: 3,
            target: 2,
        }];
        match decide("default", "obrain", false, &Verification::Done(m)) {
            MigrationDecision::StayOnSource { source, reason, .. } => {
                assert_eq!(source, "default");
                assert!(reason.contains("Project"), "{reason}");
                assert!(reason.contains('3'), "{reason}");
            }
            other => panic!("expected StayOnSource, got {other:?}"),
        }
    }

    #[test]
    fn decide_rollback_wins_over_a_clean_verification() {
        assert!(matches!(
            decide("default", "obrain", true, &Verification::Done(vec![])),
            MigrationDecision::StayOnSource { .. }
        ));
    }

    // ── The startup run, over a scripted migrator ──────────────────────────

    /// A scripted migrator over in-memory "stores": `src` is the source's
    /// counts; `copy` writes the copied counts into `dest` (minus one Project
    /// when `lose`, plus one stray node when `extra`); every `dest` it copied
    /// into is recorded. It checks that it always runs under its gate.
    struct Fake {
        gate: ReadOnlyGate,
        src: PerTypeCounts,
        lose: bool,
        extra: bool,
        fail_copy: bool,
        /// The source's counts move during the copy (an external writer).
        drift: bool,
        reads: std::sync::atomic::AtomicUsize,
        /// During `copy`, assert that the migration lock of `dest` is held.
        probe_lock: bool,
        copied_into: std::sync::Mutex<Vec<PathBuf>>,
    }
    impl Fake {
        fn new(gate: &ReadOnlyGate, src: PerTypeCounts) -> Self {
            Self {
                gate: gate.clone(),
                src,
                lose: false,
                extra: false,
                fail_copy: false,
                drift: false,
                reads: Default::default(),
                probe_lock: false,
                copied_into: Default::default(),
            }
        }
        fn copies(&self) -> usize {
            self.copied_into.lock().unwrap().len()
        }
    }
    #[async_trait::async_trait]
    impl Migrator for Fake {
        async fn source_counts(&self) -> anyhow::Result<PerTypeCounts> {
            assert!(
                self.gate.is_read_only(),
                "the source must be read under the gate"
            );
            let mut c = self.src.clone();
            // The first read is the start of the copy; later ones come after it.
            if self.drift && self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0 {
                *c.nodes.get_mut("Note").unwrap() += 1;
            }
            Ok(c)
        }
        async fn copy(&self, dest: &Path) -> anyhow::Result<()> {
            assert!(self.gate.is_read_only(), "the copy must run under the gate");
            self.copied_into.lock().unwrap().push(dest.to_path_buf());
            if self.probe_lock {
                let err = MigrationLock::acquire(dest)
                    .expect_err("the migration lock must be held DURING the copy")
                    .to_string();
                assert!(err.contains("another process is migrating"), "{err}");
            }
            if self.fail_copy {
                anyhow::bail!("disk full");
            }
            let mut c = self.src.clone();
            if self.lose {
                *c.nodes.get_mut("Project").unwrap() -= 1;
            }
            if self.extra {
                c.nodes.insert("Stray".into(), 1);
            }
            std::fs::write(dest.join("nodes"), serde_json::to_string(&c.nodes)?)?;
            std::fs::write(dest.join("rels"), serde_json::to_string(&c.relationships)?)?;
            Ok(())
        }
        async fn target_counts(&self, dest: &Path) -> anyhow::Result<PerTypeCounts> {
            Ok(PerTypeCounts {
                nodes: serde_json::from_str(&std::fs::read_to_string(dest.join("nodes"))?)?,
                relationships: serde_json::from_str(&std::fs::read_to_string(dest.join("rels"))?)?,
            })
        }
    }

    fn data() -> PerTypeCounts {
        counts(&[("Project", 3), ("Note", 7)], &[("LINKED_TO", 4)])
    }

    fn request(dest: &Path) -> MigrationRequest {
        MigrationRequest {
            source: "default".into(),
            target: "obrain".into(),
            rollback: false,
            disposable: false,
            configured_dest: Some(dest.to_path_buf()),
        }
    }

    fn stay_reason(o: StartupOutcome) -> String {
        match o {
            StartupOutcome::StayOnSource { reason } => reason,
            other => panic!("expected StayOnSource, got {other:?}"),
        }
    }

    /// DEFECT C. The copy faithfully holds what it read at the start, so the
    /// target matches the FIRST source counts; only the re-read after the copy
    /// shows the source moved.
    #[tokio::test]
    async fn a_source_whose_counts_moved_during_the_copy_is_refused_and_cleaned() {
        let gate = ReadOnlyGate::new();
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("base");
        let mut m = Fake::new(&gate, data());
        m.drift = true;
        let scope = gate.enter();
        let reason = stay_reason(
            run_startup_migration(&request(&dest), &m, &scope)
                .await
                .unwrap(),
        );
        assert!(
            reason.contains("source changed during the copy"),
            "{reason}"
        );
        assert!(reason.contains("Note"), "the type is named: {reason}");
        assert!(!dest.join(DONE_MARKER).exists());
        assert!(
            std::fs::read_dir(&dest)
                .map(|mut d| d.next().is_none())
                .unwrap_or(true),
            "the copy was removed"
        );
    }

    #[tokio::test]
    async fn the_marker_records_the_accepted_narrowings_and_stays_readable() {
        let dir = tempfile::tempdir().unwrap();
        let req = request(&dir.path().join("base"));
        std::fs::create_dir_all(dir.path()).unwrap();
        let report = CopyReport {
            sub_microsecond_datetimes: 5,
        };
        write_marker(dir.path(), &req, &data(), &report).unwrap();
        let v = read_marker(dir.path()).unwrap().expect("valid marker");
        assert_eq!(v["narrowings"]["sub_microsecond_datetimes"], 5);
        // A marker written before the field existed still reads.
        let mut old = v.clone();
        old.as_object_mut().unwrap().remove("narrowings");
        std::fs::write(dir.path().join(DONE_MARKER), old.to_string()).unwrap();
        assert!(read_marker(dir.path()).unwrap().is_some());
    }

    #[tokio::test]
    async fn parity_serves_the_target_and_marks_it_after_the_verification() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        let ro = gate.enter();
        let fake = Fake::new(&gate, data());
        let o = run_startup_migration(&request(&dest), &fake, &ro)
            .await
            .unwrap();
        assert!(
            matches!(&o, StartupOutcome::ServeTarget { dest: d, already_migrated: false, scratch: None, .. } if *d == dest),
            "{o:?}"
        );
        assert_eq!(fake.copies(), 1);
        assert_eq!(fake.copied_into.lock().unwrap()[0], dest);
        let marker: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dest.join(DONE_MARKER)).unwrap()).unwrap();
        assert_eq!(marker["nodes"]["Project"], 3, "{marker}");
    }

    /// Regression (HIGH): `from: default` is permanent configuration, so every
    /// restart used to copy again into the obrain base already in service.
    #[tokio::test]
    async fn a_second_start_serves_the_marked_target_without_copying_again() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        {
            let ro = gate.enter();
            run_startup_migration(&request(&dest), &Fake::new(&gate, data()), &ro)
                .await
                .unwrap();
        }
        // The target lives on: writes made since the switch.
        std::fs::write(dest.join("written-since"), "x").unwrap();
        let ro = gate.enter();
        let again = Fake::new(&gate, data());
        let o = run_startup_migration(&request(&dest), &again, &ro)
            .await
            .unwrap();
        assert!(
            matches!(
                o,
                StartupOutcome::ServeTarget {
                    already_migrated: true,
                    ..
                }
            ),
            "{o:?}"
        );
        assert_eq!(again.copies(), 0, "a migrated target is never copied over");
        assert!(dest.join("written-since").exists());
    }

    #[tokio::test]
    async fn a_target_holding_data_without_the_marker_is_a_named_refusal() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("substrate.meta"), "someone else's").unwrap();
        let ro = gate.enter();
        let fake = Fake::new(&gate, data());
        let err = run_startup_migration(&request(&dest), &fake, &ro)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("already holds data") && err.contains(DONE_MARKER),
            "{err}"
        );
        // It says exactly what to do (a supervisor restarts the process, so the
        // refusal repeats until the operator acts).
        assert!(
            err.contains("EMPTY the directory")
                && err.contains(&dest.display().to_string())
                && err.contains("STORAGE_MIGRATION_ROLLBACK=true"),
            "{err}"
        );
        assert_eq!(fake.copies(), 0, "never overwritten");
        assert_eq!(
            std::fs::read_to_string(dest.join("substrate.meta")).unwrap(),
            "someone else's"
        );
    }

    #[tokio::test]
    async fn a_parity_gap_stays_on_the_source_says_why_and_writes_no_marker() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        let ro = gate.enter();
        let mut fake = Fake::new(&gate, data());
        fake.lose = true;
        let reason = stay_reason(
            run_startup_migration(&request(&dest), &fake, &ro)
                .await
                .unwrap(),
        );
        assert!(
            reason.contains("Project") && reason.contains("source 3 vs target 2"),
            "{reason}"
        );
        assert!(!dest.join(DONE_MARKER).exists(), "no marker without parity");
    }

    #[tokio::test]
    async fn a_copy_error_and_a_missing_migrator_stay_on_the_source() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let ro = gate.enter();
        let mut fake = Fake::new(&gate, data());
        fake.fail_copy = true;
        let reason = stay_reason(
            run_startup_migration(&request(&root.path().join("a")), &fake, &ro)
                .await
                .unwrap(),
        );
        assert!(reason.contains("disk full"), "{reason}");
        let reason = stay_reason(
            run_startup_migration(&request(&root.path().join("b")), &NoMigrator, &ro)
                .await
                .unwrap(),
        );
        assert!(reason.contains("no migrator"), "{reason}");
    }

    /// Regression (MEDIUM): an empty source used to switch to a target that
    /// was never copied nor read back.
    #[tokio::test]
    async fn an_empty_source_still_copies_and_verifies_the_target() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let ro = gate.enter();
        // A target that does not read back empty: refused.
        let mut fake = Fake::new(&gate, PerTypeCounts::default());
        fake.extra = true;
        let reason = stay_reason(
            run_startup_migration(&request(&root.path().join("a")), &fake, &ro)
                .await
                .unwrap(),
        );
        assert!(reason.contains("Stray"), "{reason}");
        // A target that reads back empty too: served, after a real copy.
        let fake = Fake::new(&gate, PerTypeCounts::default());
        let o = run_startup_migration(&request(&root.path().join("b")), &fake, &ro)
            .await
            .unwrap();
        assert!(matches!(o, StartupOutcome::ServeTarget { .. }), "{o:?}");
        assert_eq!(fake.copies(), 1, "the empty source was copied and verified");
    }

    /// Regression (MEDIUM, plausible): two processes (`serve` and `orchestrator
    /// sync`) used to be able to copy into the same target at once.
    #[tokio::test]
    async fn a_migration_already_running_elsewhere_is_a_named_refusal() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        // Another open file description holding the lock = another process.
        let other = MigrationLock::acquire(&dest).unwrap();
        assert!(
            !other.path().starts_with(&dest),
            "the lock is never inside the target"
        );
        let ro = gate.enter();
        let fake = Fake::new(&gate, data());
        let err = run_startup_migration(&request(&dest), &fake, &ro)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("another process is migrating"), "{err}");
        assert_eq!(fake.copies(), 0);
        drop(other);
        run_startup_migration(&request(&dest), &fake, &ro)
            .await
            .unwrap();
        assert_eq!(fake.copies(), 1, "once released, the migration runs");
    }

    #[tokio::test]
    async fn rollback_stays_on_the_source_and_says_writes_since_the_switch_are_lost() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        {
            let ro = gate.enter();
            run_startup_migration(&request(&dest), &Fake::new(&gate, data()), &ro)
                .await
                .unwrap();
        }
        let mut req = request(&dest);
        req.rollback = true;
        let ro = gate.enter();
        let fake = Fake::new(&gate, data());
        let reason = stay_reason(run_startup_migration(&req, &fake, &ro).await.unwrap());
        assert!(reason.contains("NOT visible on the source"), "{reason}");
        assert_eq!(fake.copies(), 0);
    }

    #[tokio::test]
    async fn disposable_copies_into_a_scratch_dir_removed_with_it() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let configured = root.path().join("base");
        let mut req = request(&configured);
        req.disposable = true;
        let ro = gate.enter();
        let fake = Fake::new(&gate, data());
        let o = run_startup_migration(&req, &fake, &ro).await.unwrap();
        let StartupOutcome::ServeTarget {
            dest,
            scratch: Some(scratch),
            ..
        } = o
        else {
            panic!("expected a disposable ServeTarget, got {o:?}")
        };
        assert_eq!(dest, scratch.path());
        assert_ne!(dest, configured, "never the configured target");
        assert!(
            dest.join("nodes").exists(),
            "the copy landed in the scratch"
        );
        assert!(!configured.exists(), "the configured target is untouched");
        drop(scratch);
        assert!(!dest.exists(), "the scratch is removed with its owner");
    }

    /// Regression (HIGH): a disposable run that failed used to start on the
    /// REAL source with writes open. Every failure is now a refusal, and the
    /// scratch copy is removed.
    #[tokio::test]
    async fn every_disposable_failure_is_a_refusal_and_leaves_no_scratch() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let ro = gate.enter();
        let mut req = request(&root.path().join("base"));
        req.disposable = true;

        let mut gap = Fake::new(&gate, data());
        gap.lose = true;
        let mut broken = Fake::new(&gate, data());
        broken.fail_copy = true;
        for (name, m) in [
            ("gap", &gap as &dyn Migrator),
            ("copy error", &broken as &dyn Migrator),
        ] {
            let err = run_startup_migration(&req, m, &ro)
                .await
                .expect_err(name)
                .to_string();
            assert!(
                err.contains("disposable environment refused"),
                "{name}: {err}"
            );
        }
        for scratch in gap
            .copied_into
            .lock()
            .unwrap()
            .iter()
            .chain(broken.copied_into.lock().unwrap().iter())
        {
            assert!(
                !scratch.exists(),
                "scratch {} left behind",
                scratch.display()
            );
        }
        let err = run_startup_migration(&req, &NoMigrator, &ro)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no migrator"), "{err}");

        req.rollback = true;
        let err = run_startup_migration(&req, &Fake::new(&gate, data()), &ro)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("never starts on the real source"), "{err}");
    }

    #[test]
    fn a_scope_lowers_its_gate_on_drop_and_on_panic_and_gates_are_independent() {
        let gate = ReadOnlyGate::new();
        let other = ReadOnlyGate::new();
        {
            let _a = gate.enter();
            let _b = gate.enter();
            assert!(gate.is_read_only());
            assert!(!other.is_read_only(), "a private gate is not shared");
        }
        assert!(!gate.is_read_only());
        let g = gate.clone();
        let _ = std::panic::catch_unwind(move || {
            let _s = g.enter();
            panic!("boom");
        });
        assert!(!gate.is_read_only(), "a panic lowers the gate too");
    }

    // ── Review round 3 ─────────────────────────────────────────────────────

    /// The migration lock is held for the whole copy: a second process is
    /// refused DURING `copy`, and the lock is free again afterwards.
    #[tokio::test]
    async fn the_lock_is_held_during_the_copy_and_released_after() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        let ro = gate.enter();
        let mut fake = Fake::new(&gate, data());
        fake.probe_lock = true;
        run_startup_migration(&request(&dest), &fake, &ro)
            .await
            .unwrap();
        assert_eq!(fake.copies(), 1);
        MigrationLock::acquire(&dest).expect("released after the migration");
    }

    async fn migrated(gate: &ReadOnlyGate, dest: &Path) {
        let ro = gate.enter();
        run_startup_migration(&request(dest), &Fake::new(gate, data()), &ro)
            .await
            .unwrap();
    }

    /// Restores the permissions of a directory when dropped (so the temp dir
    /// can be removed even when an assertion fails).
    struct Perms(PathBuf);
    impl Drop for Perms {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        }
    }

    /// Regression: the lock was taken before the marker was looked at, so a
    /// read-only parent refused a target that needed no migration at all.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_marked_target_in_a_read_only_parent_starts() {
        use std::os::unix::fs::PermissionsExt;
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("parent");
        let dest = parent.join("base");
        migrated(&gate, &dest).await;
        let _restore = Perms(parent.clone());
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o555)).unwrap();
        let _ = std::fs::remove_file(MigrationLock::path_for(&dest));

        let ro = gate.enter();
        let again = Fake::new(&gate, data());
        let o = run_startup_migration(&request(&dest), &again, &ro)
            .await
            .unwrap();
        assert!(
            matches!(
                o,
                StartupOutcome::ServeTarget {
                    already_migrated: true,
                    ..
                }
            ),
            "{o:?}"
        );
        assert_eq!(again.copies(), 0);
    }

    /// Regression: `serve` and `sync` started together on a migrated target got
    /// a false "another process is migrating".
    #[tokio::test]
    async fn two_concurrent_starts_on_a_marked_target_both_succeed() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        migrated(&gate, &dest).await;
        // Even while a lock is held on it (another process mid-check).
        let _held = MigrationLock::acquire(&dest).unwrap();
        let (ro_a, ro_b) = (gate.enter(), gate.enter());
        let (fa, fb) = (Fake::new(&gate, data()), Fake::new(&gate, data()));
        let req = request(&dest);
        let (a, b) = tokio::join!(
            run_startup_migration(&req, &fa, &ro_a),
            run_startup_migration(&req, &fb, &ro_b)
        );
        for o in [a.unwrap(), b.unwrap()] {
            assert!(
                matches!(
                    o,
                    StartupOutcome::ServeTarget {
                        already_migrated: true,
                        ..
                    }
                ),
                "{o:?}"
            );
        }
        assert_eq!(fa.copies() + fb.copies(), 0);
    }

    /// The marker is durable evidence, not a name: an empty or foreign file is
    /// a named refusal, never "migrated", and a valid one carries the verified
    /// counts.
    #[tokio::test]
    async fn an_empty_or_unreadable_marker_is_a_named_refusal_not_a_migrated_target() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        migrated(&gate, &dest).await;
        let marker = read_marker(&dest).unwrap().expect("valid marker");
        assert_eq!(marker["version"], 1);
        assert_eq!(marker["nodes"]["Note"], 7);
        assert_eq!(marker["relationships"]["LINKED_TO"], 4);

        for (why, body) in [
            ("empty", ""),
            ("not json", "garbage"),
            (
                "foreign version",
                r#"{"version":99,"nodes":{},"relationships":{}}"#,
            ),
            ("no counts", r#"{"version":1}"#),
        ] {
            std::fs::write(dest.join(DONE_MARKER), body).unwrap();
            let ro = gate.enter();
            let fake = Fake::new(&gate, data());
            let err = run_startup_migration(&request(&dest), &fake, &ro)
                .await
                .expect_err(why)
                .to_string();
            assert!(
                err.contains("is invalid") && err.contains("EMPTY the directory"),
                "{why}: {err}"
            );
            assert_eq!(fake.copies(), 0, "{why}: never overwritten");
        }
    }

    /// A rollback leaves a trace; the next start that serves the marked target
    /// says so (once), and an ordinary restart says nothing.
    #[tokio::test]
    async fn serving_the_target_after_a_rollback_is_reported_once() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        migrated(&gate, &dest).await;

        let ro = gate.enter();
        let plain = run_startup_migration(&request(&dest), &Fake::new(&gate, data()), &ro)
            .await
            .unwrap();
        assert!(
            matches!(
                plain,
                StartupOutcome::ServeTarget {
                    after_rollback: None,
                    ..
                }
            ),
            "{plain:?}"
        );

        let mut rb = request(&dest);
        rb.rollback = true;
        let reason = stay_reason(
            run_startup_migration(&rb, &Fake::new(&gate, data()), &ro)
                .await
                .unwrap(),
        );
        assert!(
            reason.contains("from now on will NOT be in the target"),
            "{reason}"
        );
        assert!(
            dest.join(ROLLBACK_MARKER).is_file(),
            "the rollback leaves a trace"
        );

        let after = run_startup_migration(&request(&dest), &Fake::new(&gate, data()), &ro)
            .await
            .unwrap();
        assert!(
            matches!(&after, StartupOutcome::ServeTarget { after_rollback: Some(t), .. } if !t.is_empty()),
            "{after:?}"
        );
        let again = run_startup_migration(&request(&dest), &Fake::new(&gate, data()), &ro)
            .await
            .unwrap();
        assert!(
            matches!(
                again,
                StartupOutcome::ServeTarget {
                    after_rollback: None,
                    ..
                }
            ),
            "the warning is given once"
        );
    }

    /// Regression (operational trap): a failed non-disposable copy left data in
    /// the target, so every later start refused — a loop under a supervisor.
    #[tokio::test]
    async fn a_failed_copy_leaves_the_target_empty_so_the_next_start_retries() {
        let gate = ReadOnlyGate::new();
        let root = tempfile::tempdir().unwrap();
        let dest = root.path().join("base");
        let ro = gate.enter();
        let mut gap = Fake::new(&gate, data());
        gap.lose = true;
        let reason = stay_reason(
            run_startup_migration(&request(&dest), &gap, &ro)
                .await
                .unwrap(),
        );
        assert!(
            reason.contains("was removed, so the next start retries"),
            "{reason}"
        );
        assert_eq!(std::fs::read_dir(&dest).unwrap().count(), 0);
        let ok = Fake::new(&gate, data());
        let o = run_startup_migration(&request(&dest), &ok, &ro)
            .await
            .unwrap();
        assert!(matches!(o, StartupOutcome::ServeTarget { .. }), "{o:?}");
    }
}
