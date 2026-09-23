//! MaintenanceCheck — runs deep_maintenance for every project once a day.
//!
//! Wraps `skills::maintenance::deep_maintenance` as a heartbeat check:
//! aggressive cleanup (decay, energy update, staleness scoring, stuck task
//! detection, Louvain skill evolution) for each project, at most once per
//! [`PROJECT_MIN_GAP`].
//!
//! The work is split into bounded slices. A pass over all projects takes
//! ~12s per project, so a single pass over 40+ projects cannot fit in one
//! heartbeat timeout: it used to be cancelled every time, restart from the
//! first project on the next tick, and loop forever (see engine.rs
//! `TIMEOUT_RETRY_BACKOFF`). Each run now handles the projects that are due,
//! least-recently-maintained first, until [`SLICE_BUDGET`] is spent; the
//! next run resumes with the rest.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::heartbeat::{HeartbeatCheck, HeartbeatContext};
use crate::notes::NoteManager;
use crate::skills::maintenance::SkillMaintenanceConfig;
use uuid::Uuid;

/// Per-run timeout for deep maintenance.
///
/// `deep_maintenance` runs Louvain community detection + skill-evolution
/// analysis + persistence across ALL projects, which cannot complete within
/// the engine's 5s default. Without this override the run is cancelled every
/// tick and `last_run` is never updated (see engine.rs), so it retries-and-
/// times-out forever — meaning periodic skill detection/evolution never lands.
const MAINTENANCE_TIMEOUT: Duration = Duration::from_secs(5 * 60); // 5 minutes

/// Time spent starting new projects in one run. Checked before each project,
/// so it must leave room under [`MAINTENANCE_TIMEOUT`] for the last one.
const SLICE_BUDGET: Duration = Duration::from_secs(3 * 60);

/// How often a slice runs. With [`SLICE_BUDGET`] this covers ~12 projects
/// per slice, i.e. every project within a couple of hours of becoming due.
const SLICE_INTERVAL: Duration = Duration::from_secs(15 * 60);

/// Minimum time between two deep maintenances of the same project.
const PROJECT_MIN_GAP: Duration = Duration::from_secs(24 * 60 * 60);

/// Run deep maintenance on each project once per [`PROJECT_MIN_GAP`], in
/// bounded slices.
#[derive(Default)]
pub struct MaintenanceCheck {
    /// Last attempt per project, recorded BEFORE running it, so a project
    /// that times out goes to the back of the queue instead of pinning it.
    /// Seeded from the graph on the first run (`last_deep_maintenance_at`),
    /// so a restart does not re-run a full pass over every project.
    attempts: Mutex<HashMap<Uuid, Instant>>,
    /// Whether `attempts` was seeded from the graph yet.
    seeded: std::sync::atomic::AtomicBool,
}

/// Convert a persisted wall-clock time into the monotonic clock used for
/// scheduling (clamped to "now" if it lies in the future).
fn instant_from(at: chrono::DateTime<chrono::Utc>, now: Instant) -> Instant {
    let age = (chrono::Utc::now() - at).to_std().unwrap_or_default();
    now.checked_sub(age).unwrap_or(now)
}

impl MaintenanceCheck {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Projects due for maintenance at `now`, never-attempted first (in input
/// order), then least recently attempted first.
fn due_projects(
    project_ids: &[Uuid],
    attempts: &HashMap<Uuid, Instant>,
    now: Instant,
    min_gap: Duration,
) -> Vec<Uuid> {
    let mut due: Vec<(Option<Instant>, usize, Uuid)> = project_ids
        .iter()
        .enumerate()
        .filter_map(|(i, id)| match attempts.get(id) {
            Some(last) if now.duration_since(*last) < min_gap => None,
            last => Some((last.copied(), i, *id)),
        })
        .collect();
    // None < Some(_): never-attempted projects come first.
    due.sort_by_key(|(last, i, _)| (*last, *i));
    due.into_iter().map(|(_, _, id)| id).collect()
}

#[async_trait]
impl HeartbeatCheck for MaintenanceCheck {
    fn name(&self) -> &str {
        "deep_maintenance"
    }

    fn interval(&self) -> Duration {
        SLICE_INTERVAL
    }

    fn timeout_override(&self) -> Option<Duration> {
        Some(MAINTENANCE_TIMEOUT)
    }

    async fn run(&self, ctx: &HeartbeatContext) -> Result<()> {
        let projects = ctx.graph.list_projects().await?;
        let config = SkillMaintenanceConfig::default();

        // Built once and reused across projects, mirroring SynapseReplenishCheck.
        // Enables deep_maintenance's weekly step to self-heal a decayed SYNAPSE
        // graph (bounded, project-scoped) before Louvain re-detection, instead
        // of decaying synapses 3x and never repairing them — see
        // `skills::detection::ensure_synapse_graph_health`.
        let note_manager = ctx
            .search
            .clone()
            .map(|search| NoteManager::new(ctx.graph.clone(), search));
        if note_manager.is_none() {
            warn!("MaintenanceCheck: no search store available, skill self-heal disabled for this run");
        }

        if !self.seeded.swap(true, std::sync::atomic::Ordering::SeqCst) {
            match ctx.graph.get_deep_maintenance_times().await {
                Ok(times) => {
                    let now = Instant::now();
                    let mut attempts = self.attempts.lock().unwrap_or_else(|e| e.into_inner());
                    for (id, at) in times {
                        attempts.entry(id).or_insert_with(|| instant_from(at, now));
                    }
                }
                Err(e) => warn!("MaintenanceCheck: could not load last maintenance times: {e}"),
            }
        }

        let started = Instant::now();
        let due = {
            let attempts = self.attempts.lock().unwrap_or_else(|e| e.into_inner());
            let ids: Vec<Uuid> = projects.iter().map(|p| p.id).collect();
            due_projects(&ids, &attempts, started, PROJECT_MIN_GAP)
        };
        if due.is_empty() {
            debug!("MaintenanceCheck: no project due");
            return Ok(());
        }
        let by_id: HashMap<Uuid, _> = projects.iter().map(|p| (p.id, p)).collect();

        for project_id in due {
            if started.elapsed() >= SLICE_BUDGET {
                debug!("MaintenanceCheck: slice budget spent, resuming next run");
                break;
            }
            let Some(project) = by_id.get(&project_id) else {
                continue;
            };
            self.attempts
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(project.id, Instant::now());
            if let Err(e) = ctx.graph.mark_deep_maintenance(project.id).await {
                debug!("MaintenanceCheck: could not persist maintenance time: {e}");
            }

            info!(
                "MaintenanceCheck: running deep maintenance for '{}'",
                project.name
            );

            match crate::skills::maintenance::deep_maintenance(
                ctx.graph.as_ref(),
                note_manager.as_ref(),
                project.id,
                &config,
            )
            .await
            {
                Ok(report) => {
                    debug!(
                        "MaintenanceCheck: deep maintenance completed for '{}' — \
                         stale_notes_flagged: {}, stuck_tasks: {}, recommendations: {}",
                        project.name,
                        report.stale_notes_flagged,
                        report.stuck_tasks_found,
                        report.recommendations.len(),
                    );

                    // Create alert if stagnation was detected
                    if report.stagnation.is_stagnating {
                        let alert = crate::neo4j::models::AlertNode::new_for_subject(
                            "stagnation".to_string(),
                            crate::neo4j::models::AlertSeverity::Warning,
                            format!(
                                "Stagnation detected in project '{}': {} stale notes, {} stuck tasks. {}",
                                project.name,
                                report.stale_notes_flagged,
                                report.stuck_tasks_found,
                                report.recommendations.first().cloned().unwrap_or_default(),
                            ),
                            Some(project.id),
                            "project-stagnating",
                        );

                        if let Err(e) = ctx.graph.create_alert(&alert).await {
                            warn!(
                                "MaintenanceCheck: failed to create stagnation alert for '{}': {}",
                                project.name, e
                            );
                        }

                        if let Some(ref emitter) = ctx.emitter {
                            emitter.emit_created(
                                crate::events::EntityType::Alert,
                                &alert.id.to_string(),
                                serde_json::json!({
                                    "alert_type": "stagnation",
                                    "project": project.name,
                                }),
                                Some(project.id.to_string()),
                            );
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "MaintenanceCheck: deep maintenance failed for '{}': {}",
                        project.name, e
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maintenance_check_name() {
        let check = MaintenanceCheck::new();
        assert_eq!(check.name(), "deep_maintenance");
    }

    #[test]
    fn test_maintenance_check_interval() {
        let check = MaintenanceCheck::new();
        // Slices run every 15 min; each project is still maintained at most
        // once per PROJECT_MIN_GAP (see test_due_projects_*).
        assert_eq!(check.interval(), Duration::from_secs(15 * 60));
        assert!(SLICE_BUDGET < MAINTENANCE_TIMEOUT);
    }

    #[test]
    fn test_due_projects_orders_never_attempted_then_oldest() {
        let now = Instant::now();
        let (a, b, c, d) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        let mut attempts = HashMap::new();
        attempts.insert(a, now - Duration::from_secs(30 * 3600)); // due, older
        attempts.insert(b, now - Duration::from_secs(25 * 3600)); // due, newer
        attempts.insert(c, now - Duration::from_secs(3600)); // done recently
        let due = due_projects(&[a, b, c, d], &attempts, now, PROJECT_MIN_GAP);
        assert_eq!(due, vec![d, a, b]);
    }

    #[test]
    fn test_instant_from_persisted_time() {
        let now = Instant::now();
        let two_hours_ago = chrono::Utc::now() - chrono::Duration::hours(2);
        let i = instant_from(two_hours_ago, now);
        let age = now.duration_since(i);
        assert!(age >= Duration::from_secs(7190) && age <= Duration::from_secs(7210));
        // A future timestamp (clock skew) counts as "just now".
        assert_eq!(
            instant_from(chrono::Utc::now() + chrono::Duration::hours(1), now),
            now
        );
    }

    #[test]
    fn test_restart_does_not_rerun_recently_maintained_projects() {
        // Regression: attempts lived only in memory, so every restart ran a
        // full deep pass over every project again.
        // The graph read/write is covered by tests/data_migrations.rs.
        let project = crate::test_helpers::test_project_named("Recent");
        let check = MaintenanceCheck::new();
        // Seed as the graph would: maintained one hour ago.
        {
            let now = Instant::now();
            check.attempts.lock().unwrap().insert(
                project.id,
                instant_from(chrono::Utc::now() - chrono::Duration::hours(1), now),
            );
            check
                .seeded
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let attempts = check.attempts.lock().unwrap().clone();
        let due = due_projects(&[project.id], &attempts, Instant::now(), PROJECT_MIN_GAP);
        assert!(
            due.is_empty(),
            "maintained an hour ago: not due after a restart"
        );
    }

    #[test]
    fn test_due_projects_resumes_where_the_last_slice_stopped() {
        // Regression: a pass that cannot finish must not restart from the
        // first project forever. Projects attempted in the previous slice
        // are not due again; the rest are.
        let now = Instant::now();
        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        let mut attempts = HashMap::new();
        attempts.insert(ids[0], now);
        attempts.insert(ids[1], now);
        let due = due_projects(&ids, &attempts, now, PROJECT_MIN_GAP);
        assert_eq!(due, ids[2..].to_vec());
    }

    #[test]
    fn test_maintenance_check_timeout_override() {
        // Must override the engine's 5s default, otherwise deep_maintenance
        // (Louvain + evolution across all projects) is cancelled every tick and
        // last_run is never updated → periodic skill creation never lands.
        let check = MaintenanceCheck::new();
        assert_eq!(check.timeout_override(), Some(Duration::from_secs(300)));
        assert!(
            check.timeout_override().unwrap() > Duration::from_secs(5),
            "must exceed the engine default timeout"
        );
    }

    #[tokio::test]
    async fn test_maintenance_check_run_without_search_warns_but_completes() {
        // No search store available — the self-heal NoteManager cannot be
        // built, so the run must still complete (warning only, non-fatal).
        let ctx = HeartbeatContext {
            graph: std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new()),
            search: None,
            emitter: None,
        };
        let check = MaintenanceCheck::new();
        assert!(check.run(&ctx).await.is_ok());
    }

    #[tokio::test]
    async fn test_maintenance_check_run_with_search_builds_note_manager() {
        // A search store is available — the self-heal NoteManager should be
        // built (no projects exist, so deep_maintenance itself is never
        // invoked, but the construction path must not error).
        let ctx = HeartbeatContext {
            graph: std::sync::Arc::new(crate::neo4j::mock::MockGraphStore::new()),
            search: Some(std::sync::Arc::new(
                crate::meilisearch::mock::MockSearchStore::new(),
            )),
            emitter: None,
        };
        let check = MaintenanceCheck::new();
        assert!(check.run(&ctx).await.is_ok());
    }
}
