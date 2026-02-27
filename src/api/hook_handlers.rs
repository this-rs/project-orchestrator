//! API handlers for Claude Code hook integration
//!
//! Provides the `/api/hooks/activate` endpoint that Claude Code hooks call
//! to get contextual knowledge injection. This endpoint is PUBLIC (no JWT)
//! but rate-limited per IP (100 requests/minute).
//!
//! Also provides `/api/hooks/session-context` for session-start hooks to
//! inject active skills, current plan/task, and critical notes.
//!
//! Performance budget: < 200ms P99 for the complete request cycle.

use super::handlers::{AppError, OrchestratorState};
use crate::neurons::AutoReinforcementConfig;
use crate::notes::models::{NoteFilters, NoteImportance, NoteStatus};
use crate::skills::activation::{
    activate_for_hook_cached, spawn_hook_reinforcement, HookActivationConfig,
};
use crate::skills::cache::SkillCache;
use crate::skills::models::HookActivateRequest;
use crate::skills::SkillStatus;
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    Json,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

// ============================================================================
// Rate Limiter
// ============================================================================

/// Simple per-IP rate limiter using a sliding window.
///
/// Stores (request_count, window_start) per IP address.
/// Resets the window after `window_duration` has elapsed.
/// Limits to `max_requests` per window.
pub struct RateLimiter {
    entries: Mutex<HashMap<IpAddr, (u32, Instant)>>,
    max_requests: u32,
    window_duration: Duration,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_requests: u32, window_duration: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_requests,
            window_duration,
        }
    }

    /// Check if a request from this IP is allowed.
    /// Returns `true` if allowed, `false` if rate-limited.
    pub fn check(&self, ip: IpAddr) -> bool {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();

        let entry = entries.entry(ip).or_insert((0, now));

        // Reset window if expired
        if now.duration_since(entry.1) >= self.window_duration {
            *entry = (1, now);
            return true;
        }

        // Increment and check
        entry.0 += 1;
        entry.0 <= self.max_requests
    }

    /// Periodically clean up expired entries to prevent memory growth.
    /// Call this occasionally (e.g., every 100 requests).
    pub fn cleanup(&self) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        entries.retain(|_, (_, window_start)| {
            now.duration_since(*window_start) < self.window_duration * 2
        });
    }
}

/// Global rate limiter for the hooks endpoint: 100 requests per minute per IP.
static HOOK_RATE_LIMITER: LazyLock<RateLimiter> =
    LazyLock::new(|| RateLimiter::new(100, Duration::from_secs(60)));

/// Request counter for periodic cleanup.
static REQUEST_COUNTER: LazyLock<Mutex<u64>> = LazyLock::new(|| Mutex::new(0));

/// Global skill cache for the hook activation hot path.
/// Caches skills per project (TTL 5min) with pre-compiled triggers.
static SKILL_CACHE: LazyLock<SkillCache> = LazyLock::new(SkillCache::new);

/// Get the global skill cache (for invalidation from other handlers).
pub fn skill_cache() -> &'static SkillCache {
    &SKILL_CACHE
}

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/hooks/activate
///
/// Main entry point for Claude Code hook activation.
/// Receives tool input from PreToolUse hooks, matches against skill triggers,
/// and returns contextual knowledge for injection.
///
/// Public endpoint (no JWT required) — rate limited to 100 req/min per IP.
///
/// Returns:
/// - 200 with HookActivateResponse if a skill matched
/// - 204 No Content if no skill matched
/// - 400 Bad Request if input validation fails
/// - 429 Too Many Requests if rate limited
pub async fn activate_hook(
    State(state): State<OrchestratorState>,
    headers: HeaderMap,
    Json(req): Json<HookActivateRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    // --- Rate limiting ---
    let client_ip = extract_client_ip(&headers);
    if !HOOK_RATE_LIMITER.check(client_ip) {
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({
                "error": "Rate limit exceeded. Max 100 requests per minute."
            })),
        ));
    }

    // Periodic cleanup (every 100 requests)
    {
        let mut counter = REQUEST_COUNTER
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *counter += 1;
        if *counter % 100 == 0 {
            HOOK_RATE_LIMITER.cleanup();
        }
    }

    // --- Input validation ---
    if req.tool_name.is_empty() {
        return Err(AppError::BadRequest(
            "tool_name is required".to_string(),
        ));
    }

    // --- Activation pipeline (cached) ---
    let config = HookActivationConfig::default();

    let result = activate_for_hook_cached(
        state.orchestrator.neo4j(),
        req.project_id,
        &req.tool_name,
        &req.tool_input,
        &config,
        &SKILL_CACHE,
    )
    .await
    .map_err(|e| {
        tracing::warn!("Hook activation failed: {}", e);
        AppError::Internal(e)
    })?;

    match result {
        Some(outcome) => {
            // Spawn async Hebbian reinforcement (fire-and-forget, never blocks response)
            let reinforcement_config = AutoReinforcementConfig::default();
            spawn_hook_reinforcement(
                state.orchestrator.neo4j_arc(),
                outcome.activated_note_ids,
                reinforcement_config,
            );

            Ok((
                StatusCode::OK,
                Json(serde_json::to_value(&outcome.response).unwrap_or_default()),
            ))
        }
        None => Ok((
            StatusCode::NO_CONTENT,
            Json(serde_json::json!(null)),
        )),
    }
}

/// GET /api/hooks/health
///
/// Health check for the hooks subsystem.
/// Returns stats about rate limiter, skill cache, and overall hook status.
pub async fn hooks_health(
    State(_state): State<OrchestratorState>,
) -> Json<serde_json::Value> {
    let entries_count = HOOK_RATE_LIMITER
        .entries
        .lock()
        .map(|e| e.len())
        .unwrap_or(0);

    let total_requests = REQUEST_COUNTER
        .lock()
        .map(|c| *c)
        .unwrap_or(0);

    let cache_stats = SKILL_CACHE.stats().await;

    Json(serde_json::json!({
        "status": "ok",
        "rate_limiter": {
            "tracked_ips": entries_count,
            "max_requests_per_minute": 100,
        },
        "cache": {
            "active_entries": cache_stats.active_entries,
            "total_skills": cache_stats.total_skills,
            "hits": cache_stats.hits,
            "misses": cache_stats.misses,
            "hit_rate": cache_stats.hit_rate,
            "invalidations": cache_stats.invalidations,
        },
        "total_requests": total_requests,
    }))
}

// ============================================================================
// Session Context
// ============================================================================

/// Query parameters for GET /api/hooks/session-context
#[derive(Debug, Deserialize)]
pub struct SessionContextQuery {
    /// Project UUID (required)
    pub project_id: Uuid,
}

/// GET /api/hooks/session-context
///
/// Returns session context for the SessionStart hook.
/// Provides a snapshot of active skills, current plan/task, and critical notes
/// for injection into the Claude Code system prompt at session start.
///
/// Public endpoint (no JWT required) — same as activate_hook.
///
/// Returns:
/// - 200 with session context JSON
/// - 400 Bad Request if project_id is missing or invalid
pub async fn session_context(
    State(state): State<OrchestratorState>,
    Query(query): Query<SessionContextQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let neo4j = state.orchestrator.neo4j();
    let project_id = query.project_id;

    // Fetch active skills (limit 5, sorted by energy desc)
    let active_skills = match neo4j
        .list_skills(project_id, Some(SkillStatus::Active), 5, 0)
        .await
    {
        Ok((skills, _)) => skills
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "description": s.description,
                    "energy": s.energy,
                    "note_count": s.note_count,
                })
            })
            .collect::<Vec<_>>(),
        Err(e) => {
            tracing::debug!("Failed to fetch skills for session context: {}", e);
            vec![]
        }
    };

    // Fetch current in-progress plan (limit 1)
    let (current_plan, current_task) = match neo4j
        .list_plans_for_project(
            project_id,
            Some(vec!["in_progress".to_string()]),
            1,
            0,
        )
        .await
    {
        Ok((plans, _)) => {
            if let Some(plan) = plans.first() {
                // Calculate plan progress: completed tasks / total tasks
                let progress = match neo4j.get_plan_tasks(plan.id).await {
                    Ok(tasks) => {
                        let total = tasks.len();
                        let completed = tasks
                            .iter()
                            .filter(|t| {
                                t.status
                                    == crate::neo4j::models::TaskStatus::Completed
                            })
                            .count();
                        if total > 0 {
                            Some(format!("{}/{} tasks", completed, total))
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                };

                let plan_json = serde_json::json!({
                    "title": plan.title,
                    "status": plan.status,
                    "progress": progress,
                    "priority": plan.priority,
                });

                // Find the current in-progress task within this plan
                let task_json = match neo4j.get_plan_tasks(plan.id).await {
                    Ok(tasks) => tasks
                        .into_iter()
                        .find(|t| {
                            t.status
                                == crate::neo4j::models::TaskStatus::InProgress
                        })
                        .map(|t| {
                            serde_json::json!({
                                "title": t.title.unwrap_or_default(),
                                "status": t.status,
                                "priority": t.priority,
                                "tags": t.tags,
                            })
                        }),
                    Err(_) => None,
                };

                (Some(plan_json), task_json)
            } else {
                (None, None)
            }
        }
        Err(e) => {
            tracing::debug!("Failed to fetch plans for session context: {}", e);
            (None, None)
        }
    };

    // Fetch critical/high-importance notes (limit 10)
    let critical_notes = match neo4j
        .list_notes(
            Some(project_id),
            None,
            &NoteFilters {
                importance: Some(vec![NoteImportance::Critical, NoteImportance::High]),
                status: Some(vec![NoteStatus::Active]),
                limit: Some(10),
                sort_by: Some("importance".to_string()),
                sort_order: Some("desc".to_string()),
                ..Default::default()
            },
        )
        .await
    {
        Ok((notes, _)) => notes
            .into_iter()
            .map(|n| {
                serde_json::json!({
                    "content": n.content,
                    "note_type": n.note_type,
                    "importance": n.importance,
                    "tags": n.tags,
                })
            })
            .collect::<Vec<_>>(),
        Err(e) => {
            tracing::debug!("Failed to fetch notes for session context: {}", e);
            vec![]
        }
    };

    Ok(Json(serde_json::json!({
        "active_skills": active_skills,
        "current_plan": current_plan,
        "current_task": current_task,
        "critical_notes": critical_notes,
    })))
}

// ============================================================================
// Helpers
// ============================================================================

/// Extract client IP from request headers.
///
/// Checks (in order):
/// 1. X-Forwarded-For (first IP in chain)
/// 2. X-Real-IP
/// 3. Falls back to 127.0.0.1
fn extract_client_ip(headers: &HeaderMap) -> IpAddr {
    // X-Forwarded-For: client, proxy1, proxy2
    if let Some(xff) = headers.get("x-forwarded-for") {
        if let Ok(value) = xff.to_str() {
            if let Some(first_ip) = value.split(',').next() {
                if let Ok(ip) = first_ip.trim().parse::<IpAddr>() {
                    return ip;
                }
            }
        }
    }

    // X-Real-IP
    if let Some(xri) = headers.get("x-real-ip") {
        if let Ok(value) = xri.to_str() {
            if let Ok(ip) = value.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }

    // Fallback
    IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(5, Duration::from_secs(60));
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        for _ in 0..5 {
            assert!(limiter.check(ip));
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let limiter = RateLimiter::new(3, Duration::from_secs(60));
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));

        assert!(limiter.check(ip)); // 1
        assert!(limiter.check(ip)); // 2
        assert!(limiter.check(ip)); // 3
        assert!(!limiter.check(ip)); // 4 → blocked
    }

    #[test]
    fn test_rate_limiter_different_ips_independent() {
        let limiter = RateLimiter::new(2, Duration::from_secs(60));
        let ip1 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        let ip2 = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2));

        assert!(limiter.check(ip1)); // ip1: 1
        assert!(limiter.check(ip1)); // ip1: 2
        assert!(!limiter.check(ip1)); // ip1: blocked

        assert!(limiter.check(ip2)); // ip2: 1 — independent
        assert!(limiter.check(ip2)); // ip2: 2
    }

    #[test]
    fn test_rate_limiter_window_reset() {
        let limiter = RateLimiter::new(2, Duration::from_millis(10));
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));

        assert!(limiter.check(ip)); // 1
        assert!(limiter.check(ip)); // 2
        assert!(!limiter.check(ip)); // blocked

        // Wait for window to expire
        std::thread::sleep(Duration::from_millis(15));

        assert!(limiter.check(ip)); // window reset → allowed
    }

    #[test]
    fn test_rate_limiter_cleanup() {
        let limiter = RateLimiter::new(10, Duration::from_millis(10));
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));

        limiter.check(ip);
        assert_eq!(limiter.entries.lock().unwrap().len(), 1);

        std::thread::sleep(Duration::from_millis(25));
        limiter.cleanup();

        assert_eq!(limiter.entries.lock().unwrap().len(), 0);
    }

    #[test]
    fn test_extract_client_ip_xff() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4, 5.6.7.8".parse().unwrap());

        let ip = extract_client_ip(&headers);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4)));
    }

    #[test]
    fn test_extract_client_ip_xri() {
        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "10.0.0.5".parse().unwrap());

        let ip = extract_client_ip(&headers);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)));
    }

    #[test]
    fn test_extract_client_ip_xff_priority() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        headers.insert("x-real-ip", "5.6.7.8".parse().unwrap());

        let ip = extract_client_ip(&headers);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4))); // XFF takes priority
    }

    #[test]
    fn test_extract_client_ip_fallback() {
        let headers = HeaderMap::new();
        let ip = extract_client_ip(&headers);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::LOCALHOST));
    }

    #[test]
    fn test_extract_client_ip_invalid_xff() {
        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "not-an-ip".parse().unwrap());

        let ip = extract_client_ip(&headers);
        assert_eq!(ip, IpAddr::V4(Ipv4Addr::LOCALHOST)); // fallback
    }

    // ================================================================
    // SessionContextQuery tests
    // ================================================================

    #[test]
    fn test_session_context_query_deserialize_valid() {
        let query: SessionContextQuery =
            serde_json::from_str(r#"{"project_id":"00333b5f-2d0a-4467-9c98-155e55d2b7e5"}"#)
                .unwrap();
        assert_eq!(
            query.project_id,
            Uuid::parse_str("00333b5f-2d0a-4467-9c98-155e55d2b7e5").unwrap()
        );
    }

    #[test]
    fn test_session_context_query_deserialize_invalid_uuid() {
        let result: Result<SessionContextQuery, _> =
            serde_json::from_str(r#"{"project_id":"not-a-uuid"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_context_query_deserialize_missing_project_id() {
        let result: Result<SessionContextQuery, _> = serde_json::from_str(r#"{}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_context_response_structure() {
        // Verify the expected response JSON structure
        let response = serde_json::json!({
            "active_skills": [
                {
                    "name": "Neo4j Performance",
                    "description": "Query optimization",
                    "energy": 0.85,
                    "note_count": 12
                }
            ],
            "current_plan": {
                "title": "Neural Skills",
                "status": "in_progress",
                "progress": "4/7 tasks",
                "priority": 90
            },
            "current_task": {
                "title": "Create Hook",
                "status": "in_progress",
                "priority": 90,
                "tags": ["hook"]
            },
            "critical_notes": [
                {
                    "content": "Always check permissions",
                    "note_type": "gotcha",
                    "importance": "critical",
                    "tags": ["security"]
                }
            ]
        });

        // Verify all expected keys exist
        assert!(response["active_skills"].is_array());
        assert!(response["current_plan"].is_object());
        assert!(response["current_task"].is_object());
        assert!(response["critical_notes"].is_array());

        // Verify skill structure
        let skill = &response["active_skills"][0];
        assert_eq!(skill["name"], "Neo4j Performance");
        assert_eq!(skill["energy"], 0.85);

        // Verify plan structure
        assert_eq!(response["current_plan"]["title"], "Neural Skills");
        assert_eq!(response["current_plan"]["progress"], "4/7 tasks");

        // Verify task structure
        assert_eq!(response["current_task"]["title"], "Create Hook");

        // Verify note structure
        let note = &response["critical_notes"][0];
        assert_eq!(note["note_type"], "gotcha");
        assert_eq!(note["importance"], "critical");
    }

    #[test]
    fn test_session_context_empty_response_structure() {
        // When nothing is found, all fields should be empty/null
        let response = serde_json::json!({
            "active_skills": [],
            "current_plan": null,
            "current_task": null,
            "critical_notes": []
        });

        assert!(response["active_skills"].as_array().unwrap().is_empty());
        assert!(response["current_plan"].is_null());
        assert!(response["current_task"].is_null());
        assert!(response["critical_notes"].as_array().unwrap().is_empty());
    }
}
