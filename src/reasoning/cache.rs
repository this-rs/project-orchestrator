//! Ephemeral LRU cache for ReasoningTrees.
//!
//! Trees are cached by request hash (SHA-256 of the request string + optional project_id).
//! The cache is bounded by both entry count and TTL:
//! - **Max entries**: 100 (configurable via `REASONING_TREE_CACHE_SIZE`)
//! - **TTL**: 5 minutes (configurable via `REASONING_TREE_TTL`)
//!
//! # Thread safety
//!
//! Uses `tokio::sync::RwLock<LruCache>` for concurrent access.
//! Read-heavy workload (many cache lookups, rare inserts).

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lru::LruCache;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::models::ReasoningTree;

// ============================================================================
// Configuration
// ============================================================================

/// Default TTL for cached reasoning trees.
const DEFAULT_TTL_SECS: u64 = 300; // 5 minutes

/// Default maximum number of cached entries.
const DEFAULT_MAX_SIZE: usize = 100;

/// Read the TTL from environment variable or use default.
fn cache_ttl() -> Duration {
    std::env::var("REASONING_TREE_TTL")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_TTL_SECS))
}

/// Read the max size from environment variable or use default.
fn cache_max_size() -> usize {
    std::env::var("REASONING_TREE_CACHE_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_SIZE)
}

// ============================================================================
// Cache entry
// ============================================================================

/// A cached reasoning tree with its insertion timestamp.
#[derive(Debug, Clone)]
struct CacheEntry {
    tree: ReasoningTree,
    inserted_at: Instant,
}

impl CacheEntry {
    fn is_expired(&self, ttl: Duration) -> bool {
        self.inserted_at.elapsed() > ttl
    }
}

// ============================================================================
// Cache key
// ============================================================================

/// Compute a deterministic cache key from a request string and optional project_id.
///
/// Uses SHA-256 to produce a fixed-size key regardless of request length.
fn compute_cache_key(request: &str, project_id: Option<Uuid>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(request.as_bytes());
    if let Some(pid) = project_id {
        hasher.update(pid.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

// ============================================================================
// ReasoningTreeCache
// ============================================================================

/// Thread-safe LRU cache for ephemeral ReasoningTrees.
///
/// # Usage
///
/// ```rust,ignore
/// let cache = ReasoningTreeCache::new();
///
/// // Insert a tree
/// cache.insert(tree).await;
///
/// // Lookup by request
/// if let Some(tree) = cache.get("how does chat work?", None).await {
///     // Cache hit — return directly
/// }
///
/// // Invalidate after feedback
/// cache.invalidate(tree_id).await;
/// ```
#[derive(Clone)]
pub struct ReasoningTreeCache {
    inner: Arc<RwLock<LruCache<String, CacheEntry>>>,
    /// Map tree.id → cache key for invalidation by tree ID.
    id_index: Arc<RwLock<std::collections::HashMap<Uuid, String>>>,
    ttl: Duration,
}

impl ReasoningTreeCache {
    /// Create a new cache with default or env-configured settings.
    pub fn new() -> Self {
        let max_size = cache_max_size();
        Self {
            inner: Arc::new(RwLock::new(LruCache::new(
                NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::new(DEFAULT_MAX_SIZE).unwrap()),
            ))),
            id_index: Arc::new(RwLock::new(std::collections::HashMap::with_capacity(
                max_size,
            ))),
            ttl: cache_ttl(),
        }
    }

    /// Create a cache with explicit settings (for testing).
    pub fn with_config(max_size: usize, ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LruCache::new(
                NonZeroUsize::new(max_size).unwrap_or(NonZeroUsize::new(1).unwrap()),
            ))),
            id_index: Arc::new(RwLock::new(std::collections::HashMap::with_capacity(
                max_size,
            ))),
            ttl,
        }
    }

    /// Look up a cached tree by request string and optional project_id.
    ///
    /// Returns `None` if not found or expired.
    pub async fn get(&self, request: &str, project_id: Option<Uuid>) -> Option<ReasoningTree> {
        let key = compute_cache_key(request, project_id);
        let mut cache = self.inner.write().await;

        if let Some(entry) = cache.get(&key) {
            if entry.is_expired(self.ttl) {
                // Expired — remove from both caches
                let tree_id = entry.tree.id;
                cache.pop(&key);
                drop(cache);
                self.id_index.write().await.remove(&tree_id);
                None
            } else {
                Some(entry.tree.clone())
            }
        } else {
            None
        }
    }

    /// Insert a reasoning tree into the cache.
    ///
    /// If the cache is at capacity, the least recently used entry is evicted.
    pub async fn insert(&self, tree: ReasoningTree) {
        let key = compute_cache_key(&tree.request, tree.project_id);
        let tree_id = tree.id;

        let entry = CacheEntry {
            tree,
            inserted_at: Instant::now(),
        };

        let mut cache = self.inner.write().await;

        // If evicting, collect the evicted tree id while still holding the lock
        let evicted_id = if cache.len() == cache.cap().get() {
            cache.peek_lru().map(|(_, entry)| entry.tree.id)
        } else {
            None
        };

        cache.put(key.clone(), entry);
        drop(cache);

        // Now update id_index atomically: remove evicted + insert new
        let mut id_idx = self.id_index.write().await;
        if let Some(evicted_id) = evicted_id {
            id_idx.remove(&evicted_id);
        }
        id_idx.insert(tree_id, key);
    }

    /// Look up a cached tree by its UUID.
    ///
    /// Returns `None` if not found or expired. Used by `reason_feedback`
    /// to retrieve a tree for selective persistence before cache invalidation.
    pub async fn get_by_id(&self, tree_id: Uuid) -> Option<ReasoningTree> {
        let id_idx = self.id_index.read().await;
        let key = id_idx.get(&tree_id)?.clone();
        drop(id_idx);

        let mut cache = self.inner.write().await;
        if let Some(entry) = cache.get(&key) {
            if entry.is_expired(self.ttl) {
                let tid = entry.tree.id;
                cache.pop(&key);
                drop(cache);
                self.id_index.write().await.remove(&tid);
                None
            } else {
                Some(entry.tree.clone())
            }
        } else {
            None
        }
    }

    /// Find a recently cached tree for a given project (within `max_age`).
    ///
    /// Used for temporal correlation: when a note is created shortly after
    /// a reasoning tree was built for the same project, we persist the tree.
    /// Returns the most recently inserted matching tree, if any.
    pub async fn get_recent_for_project(
        &self,
        project_id: Uuid,
        max_age: Duration,
    ) -> Option<ReasoningTree> {
        let cache = self.inner.read().await;
        let mut best: Option<(&CacheEntry, Instant)> = None;

        for (_, entry) in cache.iter() {
            if entry.tree.project_id == Some(project_id)
                && entry.inserted_at.elapsed() <= max_age
                && !entry.is_expired(self.ttl)
            {
                match &best {
                    None => best = Some((entry, entry.inserted_at)),
                    Some((_, prev_ts)) if entry.inserted_at > *prev_ts => {
                        best = Some((entry, entry.inserted_at));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(entry, _)| entry.tree.clone())
    }

    /// Invalidate a cached tree by its UUID.
    ///
    /// Used after `reason_feedback` to ensure re-computation with updated scores.
    pub async fn invalidate(&self, tree_id: Uuid) -> bool {
        let mut id_idx = self.id_index.write().await;
        if let Some(key) = id_idx.remove(&tree_id) {
            let mut cache = self.inner.write().await;
            cache.pop(&key);
            true
        } else {
            false
        }
    }

    /// Invalidate all cached trees.
    pub async fn invalidate_all(&self) {
        self.inner.write().await.clear();
        self.id_index.write().await.clear();
    }

    /// Number of entries currently in the cache (including potentially expired ones).
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Check if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

impl Default for ReasoningTreeCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree(request: &str, project_id: Option<Uuid>) -> ReasoningTree {
        ReasoningTree::new(request, project_id)
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));
        let tree = make_tree("how does chat work?", None);
        let tree_id = tree.id;

        cache.insert(tree).await;

        let result = cache.get("how does chat work?", None).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, tree_id);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));

        let result = cache.get("nonexistent query", None).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_project_scoping() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));
        let project_a = Uuid::new_v4();
        let project_b = Uuid::new_v4();

        let tree_a = make_tree("same query", Some(project_a));
        let tree_b = make_tree("same query", Some(project_b));
        let id_a = tree_a.id;
        let id_b = tree_b.id;

        cache.insert(tree_a).await;
        cache.insert(tree_b).await;

        // Same request, different project → different cache entries
        let result_a = cache.get("same query", Some(project_a)).await;
        let result_b = cache.get("same query", Some(project_b)).await;

        assert_eq!(result_a.unwrap().id, id_a);
        assert_eq!(result_b.unwrap().id, id_b);
    }

    #[tokio::test]
    async fn test_ttl_expiry() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_millis(50));
        let tree = make_tree("expiring query", None);

        cache.insert(tree).await;
        assert!(cache.get("expiring query", None).await.is_some());

        // Wait for TTL
        tokio::time::sleep(Duration::from_millis(60)).await;

        assert!(cache.get("expiring query", None).await.is_none());
    }

    #[tokio::test]
    async fn test_invalidate_by_id() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));
        let tree = make_tree("to be invalidated", None);
        let tree_id = tree.id;

        cache.insert(tree).await;
        assert!(cache.get("to be invalidated", None).await.is_some());

        let removed = cache.invalidate(tree_id).await;
        assert!(removed);

        assert!(cache.get("to be invalidated", None).await.is_none());
    }

    #[tokio::test]
    async fn test_invalidate_nonexistent() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));
        let removed = cache.invalidate(Uuid::new_v4()).await;
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_invalidate_all() {
        let cache = ReasoningTreeCache::with_config(10, Duration::from_secs(60));

        cache.insert(make_tree("q1", None)).await;
        cache.insert(make_tree("q2", None)).await;
        cache.insert(make_tree("q3", None)).await;

        assert_eq!(cache.len().await, 3);

        cache.invalidate_all().await;

        assert!(cache.is_empty().await);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let cache = ReasoningTreeCache::with_config(2, Duration::from_secs(60));

        let tree1 = make_tree("query 1", None);
        let tree2 = make_tree("query 2", None);
        let tree3 = make_tree("query 3", None);

        cache.insert(tree1).await;
        cache.insert(tree2).await;

        // Cache is full (2 entries). Inserting a 3rd evicts the LRU (query 1).
        cache.insert(tree3).await;

        assert!(cache.get("query 1", None).await.is_none()); // evicted
        assert!(cache.get("query 2", None).await.is_some());
        assert!(cache.get("query 3", None).await.is_some());
    }

    #[test]
    fn test_cache_key_determinism() {
        let key1 = compute_cache_key("hello", None);
        let key2 = compute_cache_key("hello", None);
        assert_eq!(key1, key2);

        let pid = Uuid::new_v4();
        let key3 = compute_cache_key("hello", Some(pid));
        let key4 = compute_cache_key("hello", Some(pid));
        assert_eq!(key3, key4);

        // Different project → different key
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_different_requests() {
        let key1 = compute_cache_key("query A", None);
        let key2 = compute_cache_key("query B", None);
        assert_ne!(key1, key2);
    }
}
