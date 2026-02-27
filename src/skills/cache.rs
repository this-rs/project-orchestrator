//! In-memory cache for the hook activation hot path.
//!
//! Caches two expensive operations:
//! 1. **Skills per project**: avoids a Neo4j round-trip on every hook call
//! 2. **Compiled triggers**: `Regex::new()` and `glob::Pattern::new()` are
//!    compiled once and reused across requests
//!
//! TTL: 5 minutes. Bounded to 1000 entries to prevent memory leaks.
//! Invalidated on skill/note mutations via `invalidate_project()`.
//!
//! # Thread safety
//!
//! Uses `tokio::sync::RwLock<HashMap>` — read-heavy workload (many hook
//! activations, rare mutations). DashMap is not in Cargo.toml.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use regex::Regex;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::skills::models::{SkillNode, TriggerType};

// ============================================================================
// Configuration
// ============================================================================

/// Time-to-live for cached entries.
const CACHE_TTL: Duration = Duration::from_secs(300); // 5 minutes

/// Maximum number of project entries in the cache.
/// Beyond this, the oldest entries are evicted on next insert.
const MAX_ENTRIES: usize = 1000;

// ============================================================================
// Compiled triggers
// ============================================================================

/// A pre-compiled trigger pattern ready for matching without re-compilation.
#[derive(Debug, Clone)]
pub enum CompiledTrigger {
    /// Compiled regex — avoids `Regex::new()` per request (~1-5μs per compile).
    Regex(Regex),
    /// Compiled glob pattern.
    FileGlob(glob::Pattern),
    /// Semantic vector — not compiled, just stored for potential future use.
    Semantic(String),
}

/// A skill with its triggers pre-compiled for fast matching.
#[derive(Debug, Clone)]
pub struct CachedSkill {
    /// The skill node (id, name, status, etc.)
    pub skill: SkillNode,
    /// Pre-compiled triggers paired with their confidence thresholds.
    pub compiled_triggers: Vec<(CompiledTrigger, f64)>,
}

impl CachedSkill {
    /// Compile all triggers from a SkillNode.
    pub fn from_skill(skill: SkillNode) -> Self {
        let compiled_triggers = skill
            .reliable_triggers()
            .iter()
            .filter_map(|trigger| {
                let compiled = match trigger.pattern_type {
                    TriggerType::Regex => Regex::new(&trigger.pattern_value)
                        .ok()
                        .map(CompiledTrigger::Regex),
                    TriggerType::FileGlob => glob::Pattern::new(&trigger.pattern_value)
                        .ok()
                        .map(CompiledTrigger::FileGlob),
                    TriggerType::Semantic => {
                        Some(CompiledTrigger::Semantic(trigger.pattern_value.clone()))
                    }
                };
                compiled.map(|c| (c, trigger.confidence_threshold))
            })
            .collect();

        Self {
            skill,
            compiled_triggers,
        }
    }
}

// ============================================================================
// Cache entry
// ============================================================================

/// Cached skills for a single project.
struct CacheEntry {
    /// Pre-compiled matchable skills for this project.
    skills: Vec<CachedSkill>,
    /// When this entry was cached.
    cached_at: Instant,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        self.cached_at.elapsed() >= CACHE_TTL
    }
}

// ============================================================================
// SkillCache
// ============================================================================

/// In-memory cache for skill data used in the hook activation hot path.
///
/// Thread-safe via `tokio::sync::RwLock`. Optimized for read-heavy workloads
/// (many hook activations per second, rare skill/note mutations).
///
/// # Performance metrics
///
/// Tracks hits, misses, and invalidations via atomic counters for the
/// `/api/hooks/health` endpoint.
pub struct SkillCache {
    entries: RwLock<HashMap<Uuid, CacheEntry>>,
    // --- Metrics (lock-free) ---
    hits: AtomicU64,
    misses: AtomicU64,
    invalidations: AtomicU64,
}

impl SkillCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    /// Get cached skills for a project.
    ///
    /// Returns `None` on cache miss or TTL expiry.
    /// Increments hit/miss counters for metrics.
    pub async fn get(&self, project_id: &Uuid) -> Option<Vec<CachedSkill>> {
        let entries = self.entries.read().await;
        match entries.get(project_id) {
            Some(entry) if !entry.is_expired() => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.skills.clone())
            }
            _ => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert skills into the cache for a project.
    ///
    /// Skills are filtered to matchable-only and triggers are pre-compiled.
    /// If the cache exceeds `MAX_ENTRIES`, the oldest entries are evicted.
    pub async fn insert(&self, project_id: Uuid, skills: Vec<SkillNode>) {
        let cached_skills: Vec<CachedSkill> = skills
            .into_iter()
            .filter(|s| s.is_matchable())
            .map(CachedSkill::from_skill)
            .collect();

        let mut entries = self.entries.write().await;

        // Evict oldest entries if at capacity
        if entries.len() >= MAX_ENTRIES && !entries.contains_key(&project_id) {
            self.evict_oldest(&mut entries);
        }

        entries.insert(
            project_id,
            CacheEntry {
                skills: cached_skills,
                cached_at: Instant::now(),
            },
        );
    }

    /// Invalidate cache for a specific project.
    ///
    /// Call this when skills, triggers, or member notes are mutated.
    pub async fn invalidate_project(&self, project_id: &Uuid) {
        let mut entries = self.entries.write().await;
        if entries.remove(project_id).is_some() {
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Invalidate all cached entries.
    ///
    /// Call this on bulk operations like `detect_skills`.
    pub async fn invalidate_all(&self) {
        let mut entries = self.entries.write().await;
        let count = entries.len() as u64;
        entries.clear();
        self.invalidations.fetch_add(count, Ordering::Relaxed);
    }

    /// Get cache statistics for the health endpoint.
    pub async fn stats(&self) -> CacheStats {
        let entries = self.entries.read().await;
        let active_entries = entries.values().filter(|e| !e.is_expired()).count();
        let total_skills: usize = entries
            .values()
            .filter(|e| !e.is_expired())
            .map(|e| e.skills.len())
            .sum();

        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        CacheStats {
            active_entries,
            total_skills,
            hits,
            misses,
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
            invalidations: self.invalidations.load(Ordering::Relaxed),
        }
    }

    /// Evict the oldest entry to make room for a new one.
    fn evict_oldest(&self, entries: &mut HashMap<Uuid, CacheEntry>) {
        if let Some((&oldest_key, _)) = entries
            .iter()
            .min_by_key(|(_, entry)| entry.cached_at)
        {
            entries.remove(&oldest_key);
        }
    }
}

impl Default for SkillCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stats
// ============================================================================

/// Cache statistics for monitoring via `/api/hooks/health`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheStats {
    /// Number of non-expired project entries in the cache.
    pub active_entries: usize,
    /// Total number of cached skills across all projects.
    pub total_skills: usize,
    /// Number of cache hits since startup.
    pub hits: u64,
    /// Number of cache misses since startup.
    pub misses: u64,
    /// Hit rate (0.0 - 1.0).
    pub hit_rate: f64,
    /// Number of invalidations since startup.
    pub invalidations: u64,
}

// ============================================================================
// Trigger matching (using compiled triggers)
// ============================================================================

/// Evaluate a skill's compiled triggers against pattern and file context.
///
/// This is the cached equivalent of `evaluate_skill_match()` in `activation.rs`,
/// but uses pre-compiled Regex/Glob patterns instead of recompiling each time.
///
/// Returns the highest confidence score across all matching triggers.
pub fn evaluate_cached_skill(
    cached_skill: &CachedSkill,
    pattern: Option<&str>,
    file_context: Option<&str>,
) -> f64 {
    let mut max_confidence = 0.0_f64;

    for (trigger, _threshold) in &cached_skill.compiled_triggers {
        let confidence = match trigger {
            CompiledTrigger::Regex(re) => {
                if let Some(pat) = pattern {
                    if re.is_match(pat) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            CompiledTrigger::FileGlob(glob_pat) => {
                let target = file_context.or(pattern);
                if let Some(file) = target {
                    if glob_pat.matches(file) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            CompiledTrigger::Semantic(_) => {
                // Semantic matching skipped in hot path
                0.0
            }
        };

        max_confidence = max_confidence.max(confidence);
    }

    max_confidence
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_skill(name: &str, triggers: Vec<(TriggerType, &str)>) -> SkillNode {
        use crate::skills::models::SkillTrigger;

        let mut skill = SkillNode::new(Uuid::new_v4(), name);
        skill.trigger_patterns = triggers
            .into_iter()
            .map(|(tt, val)| match tt {
                TriggerType::Regex => SkillTrigger::regex(val, 0.5),
                TriggerType::FileGlob => SkillTrigger::file_glob(val, 0.5),
                TriggerType::Semantic => SkillTrigger::semantic(val, 0.5),
            })
            .collect();
        skill
    }

    #[tokio::test]
    async fn test_cache_miss_on_empty() {
        let cache = SkillCache::new();
        let project_id = Uuid::new_v4();
        assert!(cache.get(&project_id).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_hit_after_insert() {
        let cache = SkillCache::new();
        let project_id = Uuid::new_v4();
        let skills = vec![make_test_skill("Neo4j", vec![(TriggerType::Regex, "neo4j")])];

        cache.insert(project_id, skills).await;
        let result = cache.get(&project_id).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_cache_miss_after_ttl() {
        // Use a custom cache with very short TTL (simulated by manual expiry check)
        let cache = SkillCache::new();
        let project_id = Uuid::new_v4();
        let skills = vec![make_test_skill("Test", vec![(TriggerType::Regex, "test")])];

        cache.insert(project_id, skills).await;

        // Manually expire the entry
        {
            let mut entries = cache.entries.write().await;
            if let Some(entry) = entries.get_mut(&project_id) {
                entry.cached_at = Instant::now() - Duration::from_secs(600);
            }
        }

        assert!(cache.get(&project_id).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_invalidate_project() {
        let cache = SkillCache::new();
        let project_id = Uuid::new_v4();
        let skills = vec![make_test_skill("Test", vec![(TriggerType::Regex, "test")])];

        cache.insert(project_id, skills).await;
        assert!(cache.get(&project_id).await.is_some());

        cache.invalidate_project(&project_id).await;
        assert!(cache.get(&project_id).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_invalidate_all() {
        let cache = SkillCache::new();
        let p1 = Uuid::new_v4();
        let p2 = Uuid::new_v4();

        cache
            .insert(p1, vec![make_test_skill("S1", vec![(TriggerType::Regex, "a")])])
            .await;
        cache
            .insert(p2, vec![make_test_skill("S2", vec![(TriggerType::Regex, "b")])])
            .await;

        cache.invalidate_all().await;
        assert!(cache.get(&p1).await.is_none());
        assert!(cache.get(&p2).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = SkillCache::new();
        let project_id = Uuid::new_v4();

        // Miss
        cache.get(&project_id).await;

        // Insert and hit
        cache
            .insert(
                project_id,
                vec![make_test_skill("Test", vec![(TriggerType::Regex, "test")])],
            )
            .await;
        cache.get(&project_id).await;

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(stats.active_entries, 1);
        assert_eq!(stats.total_skills, 1);
    }

    #[tokio::test]
    async fn test_cache_eviction_at_capacity() {
        let cache = SkillCache::new();

        // Insert MAX_ENTRIES + 1 entries
        for i in 0..=MAX_ENTRIES {
            let project_id = Uuid::from_u128(i as u128);
            cache
                .insert(
                    project_id,
                    vec![make_test_skill(
                        &format!("Skill_{}", i),
                        vec![(TriggerType::Regex, "test")],
                    )],
                )
                .await;
        }

        let entries = cache.entries.read().await;
        assert!(entries.len() <= MAX_ENTRIES);
    }

    #[test]
    fn test_compiled_trigger_regex() {
        let skill = make_test_skill("Neo4j", vec![(TriggerType::Regex, "neo4j|cypher")]);
        let cached = CachedSkill::from_skill(skill);

        assert_eq!(cached.compiled_triggers.len(), 1);
        assert!(matches!(&cached.compiled_triggers[0].0, CompiledTrigger::Regex(_)));
    }

    #[test]
    fn test_compiled_trigger_file_glob() {
        let skill = make_test_skill("API", vec![(TriggerType::FileGlob, "src/api/**")]);
        let cached = CachedSkill::from_skill(skill);

        assert_eq!(cached.compiled_triggers.len(), 1);
        assert!(matches!(
            &cached.compiled_triggers[0].0,
            CompiledTrigger::FileGlob(_)
        ));
    }

    #[test]
    fn test_compiled_trigger_invalid_regex_skipped() {
        let skill = make_test_skill("Bad", vec![(TriggerType::Regex, "[invalid")]);
        let cached = CachedSkill::from_skill(skill);

        // Invalid regex should be skipped (filter_map returns None)
        assert_eq!(cached.compiled_triggers.len(), 0);
    }

    #[test]
    fn test_evaluate_cached_skill_regex_match() {
        let skill = make_test_skill("Neo4j", vec![(TriggerType::Regex, "neo4j|cypher")]);
        let cached = CachedSkill::from_skill(skill);

        let confidence = evaluate_cached_skill(&cached, Some("neo4j_client"), None);
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_cached_skill_regex_no_match() {
        let skill = make_test_skill("Neo4j", vec![(TriggerType::Regex, "neo4j|cypher")]);
        let cached = CachedSkill::from_skill(skill);

        let confidence = evaluate_cached_skill(&cached, Some("api_handler"), None);
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_cached_skill_file_glob_match() {
        let skill = make_test_skill("API", vec![(TriggerType::FileGlob, "src/api/**")]);
        let cached = CachedSkill::from_skill(skill);

        let confidence = evaluate_cached_skill(&cached, None, Some("src/api/handlers.rs"));
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_cached_skill_multiple_triggers_best_wins() {
        let skill = make_test_skill(
            "Mixed",
            vec![
                (TriggerType::Regex, "neo4j"),
                (TriggerType::FileGlob, "src/other/**"),
            ],
        );
        let cached = CachedSkill::from_skill(skill);

        // Regex matches, glob doesn't
        let confidence = evaluate_cached_skill(&cached, Some("neo4j_query"), Some("src/api/test.rs"));
        assert!((confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluate_cached_skill_semantic_skipped() {
        let skill = make_test_skill("Semantic", vec![(TriggerType::Semantic, "[0.1, 0.2]")]);
        let cached = CachedSkill::from_skill(skill);

        let confidence = evaluate_cached_skill(&cached, Some("anything"), None);
        assert!((confidence - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_cache_filters_non_matchable_skills() {
        let cache = SkillCache::new();
        let project_id = Uuid::new_v4();

        let mut dormant_skill = make_test_skill("Dormant", vec![(TriggerType::Regex, "test")]);
        dormant_skill.status = crate::skills::models::SkillStatus::Dormant;

        let active_skill = make_test_skill("Active", vec![(TriggerType::Regex, "test")]);

        cache
            .insert(project_id, vec![dormant_skill, active_skill])
            .await;

        let cached = cache.get(&project_id).await.unwrap();
        // Only the Active skill should be cached (Dormant is not matchable)
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].skill.name, "Active");
    }
}
