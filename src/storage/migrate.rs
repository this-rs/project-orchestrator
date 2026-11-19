//! Startup migration gate — the pieces PO itself owns for the source → target
//! storage-backend switch.
//!
//! The data copy (reading Neo4j, writing obrain) is the migrator's job and
//! lives outside this repository (the spike-migrateur). What PO owns, and what
//! this module provides, is the *guarding* and the *decision*:
//!
//! - a process-wide **read-only gate** ([`set_read_only`] / [`is_read_only`]):
//!   while a migration is in progress every write route (HTTP and MCP) must
//!   refuse with a named answer, so the source is never modified through PO;
//! - the **per-type verification** ([`PerTypeCounts`] + [`verify_by_type`]): the
//!   switch is only allowed once the source and the target agree, type by type
//!   (nodes and links);
//! - the **decision** ([`decide`]): flip to the target, or stay on the source
//!   with a named reason (rollback demanded by configuration, or a per-type
//!   mismatch).
//!
//! Everything here is pure and backend-agnostic: it is tested against in-memory
//! counts, never against a live store.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

/// The stable `code` a write refused during migration answers with, in the same
/// shape as the other named refusals of the API (`{ "code": … }`).
pub const READ_ONLY_CODE: &str = "migration_read_only";

/// The HTTP status of a write refused during migration: the client was not at
/// fault, the server is mid-switch (503 Service Unavailable).
pub const READ_ONLY_STATUS: u16 = 503;

// ── Process-wide read-only gate ────────────────────────────────────────────
//
// One gate per process: the switch happens once, at startup, before any route
// is served. A single `AtomicBool` (not a per-`ServerState` field) keeps the
// gate observable from the router middleware, the MCP dispatch and the startup
// orchestrator without threading a new field through every state constructor.

static READ_ONLY: AtomicBool = AtomicBool::new(false);

/// Tests that touch the process-wide gate serialize on this lock.
#[cfg(test)]
pub(crate) static GATE_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Enter (`true`) or leave (`false`) the read-only migration state.
pub fn set_read_only(value: bool) {
    READ_ONLY.store(value, Ordering::SeqCst);
}

/// Whether the process is currently read-only: every write route must refuse
/// with [`READ_ONLY_CODE`].
pub fn is_read_only() -> bool {
    READ_ONLY.load(Ordering::SeqCst)
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
    /// True when every label and every relationship type carries a count on
    /// both sides (i.e. no side is empty). Useful as a cheap guard against
    /// comparing a populated source to a target that was never written.
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

/// The data-copy seam. The real migrator (reading Neo4j, writing obrain) lives
/// outside this repository; PO drives it through this trait and owns the gate
/// and the decision. An implementation READS the source and WRITES only to
/// `dest`: the source is never modified.
#[async_trait::async_trait]
pub trait Migrator: Send + Sync {
    /// Per-type counts of the source.
    async fn source_counts(&self) -> anyhow::Result<PerTypeCounts>;
    /// Copy the source into `dest` (a directory owned by the target).
    async fn copy(&self, dest: &std::path::Path) -> anyhow::Result<()>;
    /// Per-type counts of what `dest` now holds, read back from the target.
    async fn target_counts(&self, dest: &std::path::Path) -> anyhow::Result<PerTypeCounts>;
}

/// The migrator wired in a build that carries none: every step fails by name,
/// so a migration request fails CLOSED (stay on the source) instead of
/// flipping blind.
pub struct NoMigrator;

#[async_trait::async_trait]
impl Migrator for NoMigrator {
    async fn source_counts(&self) -> anyhow::Result<PerTypeCounts> {
        anyhow::bail!(
            "no migrator is wired in this build (the data copy lives outside this repository)"
        )
    }
    async fn copy(&self, _dest: &std::path::Path) -> anyhow::Result<()> {
        anyhow::bail!("no migrator is wired in this build")
    }
    async fn target_counts(&self, _dest: &std::path::Path) -> anyhow::Result<PerTypeCounts> {
        anyhow::bail!("no migrator is wired in this build")
    }
}

/// Holds the process read-only for its lifetime: `enter` raises the gate, drop
/// lowers it (also on an early return or a panic). The factory keeps it until
/// the chosen backend is built, i.e. before any route or background writer of
/// `start_server` exists.
pub struct ReadOnlyScope(());

impl ReadOnlyScope {
    pub fn enter() -> Self {
        set_read_only(true);
        Self(())
    }
}

impl Drop for ReadOnlyScope {
    fn drop(&mut self) {
        set_read_only(false);
    }
}

/// The result of the startup migration: the decision, and where the target
/// lives (the configured path, or a throwaway copy when `disposable`).
#[derive(Debug)]
pub struct StartupMigration {
    pub decision: MigrationDecision,
    /// The directory holding the migrated copy; `None` when nothing was copied
    /// (rollback, empty source, or a failure before the copy).
    pub dest: Option<std::path::PathBuf>,
}

/// Run the whole startup migration: caller MUST already hold a
/// [`ReadOnlyScope`]. Reads the source counts; if the source holds data, copies
/// it to the destination (a fresh scratch directory when `disposable`, else
/// `configured_dest`), reads the target back and verifies per type. Every error
/// path is a `StayOnSource` with its reason; only a clean parity flips.
pub async fn run_startup_migration(
    source: &str,
    target: &str,
    rollback: bool,
    disposable: bool,
    configured_dest: Option<std::path::PathBuf>,
    migrator: &dyn Migrator,
) -> StartupMigration {
    let stay = |reason: String, dest: Option<std::path::PathBuf>| StartupMigration {
        decision: decide(source, target, rollback, &Verification::NotRun(reason)),
        dest,
    };
    if rollback {
        return stay("rollback".to_string(), None);
    }
    let src_counts = match migrator.source_counts().await {
        Ok(c) => c,
        Err(e) => return stay(format!("source unreadable: {e}"), None),
    };
    // A scratch destination for the disposable environment: a COPY, never the
    // source, and never the configured (real) target location.
    let dest = if disposable {
        match tempfile::Builder::new().prefix("po-disposable-").tempdir() {
            #[allow(deprecated)]
            Ok(dir) => dir.into_path(),
            Err(e) => return stay(format!("cannot create the disposable directory: {e}"), None),
        }
    } else {
        match configured_dest {
            Some(d) => d,
            None => return stay("no destination configured for the target".to_string(), None),
        }
    };
    if src_counts.is_empty() {
        // Nothing to carry over: the target starts empty, parity is trivial.
        return StartupMigration {
            decision: decide(source, target, rollback, &Verification::Done(Vec::new())),
            dest: Some(dest),
        };
    }
    if let Err(e) = migrator.copy(&dest).await {
        return stay(format!("copy failed: {e}"), Some(dest));
    }
    let tgt_counts = match migrator.target_counts(&dest).await {
        Ok(c) => c,
        Err(e) => return stay(format!("target unreadable: {e}"), Some(dest)),
    };
    StartupMigration {
        decision: decide(
            source,
            target,
            rollback,
            &Verification::Done(verify_by_type(&src_counts, &tgt_counts)),
        ),
        dest: Some(dest),
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
    fn empty_counts_are_not_empty_guard() {
        assert!(PerTypeCounts::default().is_empty());
        assert!(!counts(&[("Project", 1)], &[]).is_empty());
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
        // The case the old `verify_by_type=false` flag used to wave through.
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

    /// A scripted migrator over in-memory "stores": `src` is the source's
    /// counts; `copy` writes a marker file in `dest` and records what it saw;
    /// `lose` drops one node from the copy (an injected parity gap).
    struct Fake {
        src: PerTypeCounts,
        lose: bool,
        fail_copy: bool,
        copies: std::sync::atomic::AtomicUsize,
    }
    impl Fake {
        fn new(src: PerTypeCounts) -> Self {
            Self {
                src,
                lose: false,
                fail_copy: false,
                copies: Default::default(),
            }
        }
    }
    #[async_trait::async_trait]
    impl Migrator for Fake {
        async fn source_counts(&self) -> anyhow::Result<PerTypeCounts> {
            assert!(is_read_only(), "the source must be read under the gate");
            Ok(self.src.clone())
        }
        async fn copy(&self, dest: &std::path::Path) -> anyhow::Result<()> {
            assert!(is_read_only(), "the copy must run under the gate");
            self.copies.fetch_add(1, Ordering::SeqCst);
            if self.fail_copy {
                anyhow::bail!("disk full");
            }
            let mut c = self.src.clone();
            if self.lose {
                *c.nodes.get_mut("Project").unwrap() -= 1;
            }
            std::fs::write(
                dest.join("copy.json"),
                format!("{:?}", (&c.nodes, &c.relationships)),
            )?;
            std::fs::write(dest.join("nodes"), serde_json::to_string(&c.nodes)?)?;
            std::fs::write(dest.join("rels"), serde_json::to_string(&c.relationships)?)?;
            Ok(())
        }
        async fn target_counts(&self, dest: &std::path::Path) -> anyhow::Result<PerTypeCounts> {
            Ok(PerTypeCounts {
                nodes: serde_json::from_str(&std::fs::read_to_string(dest.join("nodes"))?)?,
                relationships: serde_json::from_str(&std::fs::read_to_string(dest.join("rels"))?)?,
            })
        }
    }

    fn data() -> PerTypeCounts {
        counts(&[("Project", 3), ("Note", 7)], &[("LINKED_TO", 4)])
    }

    #[tokio::test]
    async fn startup_migration_flips_on_parity_and_never_touches_the_source() {
        let _lock = GATE_TEST_LOCK.lock().await;
        let dest = tempfile::tempdir().unwrap();
        let _ro = ReadOnlyScope::enter();
        let fake = Fake::new(data());
        let r = run_startup_migration(
            "default",
            "obrain",
            false,
            false,
            Some(dest.path().into()),
            &fake,
        )
        .await;
        assert_eq!(r.decision, flip());
        assert_eq!(r.dest.as_deref(), Some(dest.path()));
        assert_eq!(fake.src, data(), "the source counts are unchanged");
    }

    #[tokio::test]
    async fn startup_migration_stays_on_source_on_a_parity_gap_and_says_why() {
        let _lock = GATE_TEST_LOCK.lock().await;
        let dest = tempfile::tempdir().unwrap();
        let _ro = ReadOnlyScope::enter();
        let mut fake = Fake::new(data());
        fake.lose = true;
        let r = run_startup_migration(
            "default",
            "obrain",
            false,
            false,
            Some(dest.path().into()),
            &fake,
        )
        .await;
        match r.decision {
            MigrationDecision::StayOnSource { reason, .. } => {
                assert!(
                    reason.contains("Project") && reason.contains("source 3 vs target 2"),
                    "{reason}"
                )
            }
            other => panic!("expected StayOnSource, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn startup_migration_fails_closed_on_copy_error_and_without_migrator() {
        let _lock = GATE_TEST_LOCK.lock().await;
        let dest = tempfile::tempdir().unwrap();
        let _ro = ReadOnlyScope::enter();
        let mut fake = Fake::new(data());
        fake.fail_copy = true;
        let r = run_startup_migration(
            "default",
            "obrain",
            false,
            false,
            Some(dest.path().into()),
            &fake,
        )
        .await;
        assert!(
            matches!(&r.decision, MigrationDecision::StayOnSource { reason, .. } if reason.contains("disk full"))
        );
        let r = run_startup_migration(
            "default",
            "obrain",
            false,
            false,
            Some(dest.path().into()),
            &NoMigrator,
        )
        .await;
        assert!(
            matches!(&r.decision, MigrationDecision::StayOnSource { reason, .. } if reason.contains("no migrator"))
        );
    }

    #[tokio::test]
    async fn disposable_copies_into_a_scratch_dir_not_the_configured_target() {
        let _lock = GATE_TEST_LOCK.lock().await;
        let configured = tempfile::tempdir().unwrap();
        let _ro = ReadOnlyScope::enter();
        let fake = Fake::new(data());
        let r = run_startup_migration(
            "default",
            "obrain",
            false,
            true,
            Some(configured.path().into()),
            &fake,
        )
        .await;
        assert_eq!(r.decision, flip());
        let d = r.dest.unwrap();
        assert_ne!(
            d,
            configured.path(),
            "a disposable run never uses the configured target"
        );
        assert!(
            d.join("nodes").exists(),
            "the copy landed in the scratch dir"
        );
        assert!(
            !configured.path().join("nodes").exists(),
            "the configured target is untouched"
        );
        std::fs::remove_dir_all(d).unwrap();
    }

    #[tokio::test]
    async fn read_only_scope_is_released_on_drop() {
        let _lock = GATE_TEST_LOCK.lock().await;
        {
            let _ro = ReadOnlyScope::enter();
            assert!(is_read_only());
        }
        assert!(!is_read_only());
    }
}
