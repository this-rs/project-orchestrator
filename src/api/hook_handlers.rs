//! API handlers for Claude Code hook integration
//!
//! Provides the `/api/hooks/activate` endpoint that Claude Code hooks call
//! to get contextual knowledge injection. This endpoint is PUBLIC (no JWT)
//! but rate-limited per IP (100 requests/minute).
//!
//! Performance budget: < 200ms P99 for the complete request cycle.

use super::handlers::{AppError, OrchestratorState};
use crate::neurons::AutoReinforcementConfig;
use crate::skills::activation::{
    activate_for_hook_cached, spawn_hook_reinforcement, HookActivationConfig,
};
use crate::skills::cache::SkillCache;
use crate::skills::models::HookActivateRequest;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

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
}
