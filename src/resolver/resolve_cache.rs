//! ResolveCache: LRU cache for import resolution results
//!
//! Avoids re-resolving the same import path from different source files.
//! Key: `(source_file, import_path)`, Value: `Vec<String>` (resolved paths).
//!
//! Return semantics:
//! - `get() → None`: not in cache (cache miss)
//! - `get() → Some(&vec![])`: cached as unresolvable (negative cache)
//! - `get() → Some(&vec![...])`: cached as resolved to one or more paths

use lru::LruCache;
use std::num::NonZeroUsize;

/// Default cache capacity (100K entries)
const DEFAULT_CAPACITY: usize = 100_000;

/// LRU cache for import resolution results.
///
/// Shared during a sync pass, cleared between syncs.
pub struct ResolveCache {
    cache: LruCache<(String, String), Vec<String>>,
    hits: u64,
    misses: u64,
}

/// Cache statistics
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveCacheStats {
    /// Current number of entries
    pub size: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Cache hit count
    pub hits: u64,
    /// Cache miss count
    pub misses: u64,
    /// Hit rate as a ratio (0.0 to 1.0)
    pub hit_rate: f64,
}

impl ResolveCache {
    /// Create a new resolve cache with default capacity (100K entries).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new resolve cache with a specific capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap()),
            ),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached resolution result.
    ///
    /// Returns:
    /// - `None` — not in cache (miss)
    /// - `Some(&vec![])` — cached as unresolvable (negative cache)
    /// - `Some(&vec![...])` — cached as resolved to one or more paths
    pub fn get(&mut self, source: &str, import: &str) -> Option<&Vec<String>> {
        let key = (source.to_string(), import.to_string());
        if let Some(result) = self.cache.get(&key) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a resolution result into the cache.
    ///
    /// Pass an empty Vec to cache a negative result (unresolvable import).
    /// If the cache is at capacity, the least recently used entry is evicted.
    pub fn insert(&mut self, source: String, import: String, result: Vec<String>) {
        self.cache.put((source, import), result);
    }

    /// Clear all entries and reset hit/miss counters.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> ResolveCacheStats {
        let total = self.hits + self.misses;
        ResolveCacheStats {
            size: self.cache.len(),
            capacity: self.cache.cap().get(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: if total > 0 {
                self.hits as f64 / total as f64
            } else {
                0.0
            },
        }
    }

    /// Get current number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for ResolveCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ResolveCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ResolveCache")
            .field("size", &stats.size)
            .field("capacity", &stats.capacity)
            .field("hit_rate", &format!("{:.1}%", stats.hit_rate * 100.0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_resolved() {
        let mut cache = ResolveCache::new();
        cache.insert(
            "src/main.rs".to_string(),
            "crate::api::handlers".to_string(),
            vec!["src/api/handlers.rs".to_string()],
        );

        let result = cache.get("src/main.rs", "crate::api::handlers");
        assert_eq!(result, Some(&vec!["src/api/handlers.rs".to_string()]));
    }

    #[test]
    fn test_insert_and_get_multiple() {
        let mut cache = ResolveCache::new();
        cache.insert(
            "src/Main.java".to_string(),
            "com.example.utils.*".to_string(),
            vec![
                "src/com/example/utils/StringUtils.java".to_string(),
                "src/com/example/utils/DateUtils.java".to_string(),
            ],
        );

        let result = cache.get("src/Main.java", "com.example.utils.*");
        assert_eq!(
            result,
            Some(&vec![
                "src/com/example/utils/StringUtils.java".to_string(),
                "src/com/example/utils/DateUtils.java".to_string(),
            ])
        );
    }

    #[test]
    fn test_insert_and_get_unresolvable() {
        let mut cache = ResolveCache::new();
        cache.insert(
            "src/main.rs".to_string(),
            "external::crate".to_string(),
            vec![],
        );

        let result = cache.get("src/main.rs", "external::crate");
        assert_eq!(result, Some(&vec![])); // Cached as unresolvable (empty Vec)
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ResolveCache::new();
        let result = cache.get("src/main.rs", "not::cached");
        assert_eq!(result, None); // Not in cache
    }

    #[test]
    fn test_hit_miss_tracking() {
        let mut cache = ResolveCache::new();
        cache.insert(
            "a.rs".to_string(),
            "foo".to_string(),
            vec!["foo.rs".to_string()],
        );

        let _ = cache.get("a.rs", "foo"); // hit
        let _ = cache.get("a.rs", "foo"); // hit
        let _ = cache.get("a.rs", "bar"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut cache = ResolveCache::with_capacity(3);

        cache.insert("a.rs".to_string(), "1".to_string(), vec!["r1".to_string()]);
        cache.insert("a.rs".to_string(), "2".to_string(), vec!["r2".to_string()]);
        cache.insert("a.rs".to_string(), "3".to_string(), vec!["r3".to_string()]);
        assert_eq!(cache.len(), 3);

        // Adding a 4th should evict the LRU (key "1")
        cache.insert("a.rs".to_string(), "4".to_string(), vec!["r4".to_string()]);
        assert_eq!(cache.len(), 3);

        // "1" should be evicted
        assert_eq!(cache.get("a.rs", "1"), None);
        // "2", "3", "4" should still be present
        assert!(cache.get("a.rs", "2").is_some());
        assert!(cache.get("a.rs", "3").is_some());
        assert!(cache.get("a.rs", "4").is_some());
    }

    #[test]
    fn test_large_capacity_eviction() {
        let cap = 1_000;
        let mut cache = ResolveCache::with_capacity(cap);

        // Insert cap+100 entries
        for i in 0..cap + 100 {
            cache.insert(
                "src/file.rs".to_string(),
                format!("import_{}", i),
                vec![format!("resolved_{}", i)],
            );
        }

        // Size should never exceed capacity
        assert_eq!(cache.len(), cap);

        // First 100 entries should be evicted
        assert_eq!(cache.get("src/file.rs", "import_0"), None);
        assert_eq!(cache.get("src/file.rs", "import_99"), None);

        // Last entry should be present
        assert!(cache
            .get("src/file.rs", &format!("import_{}", cap + 99))
            .is_some());
    }

    #[test]
    fn test_clear() {
        let mut cache = ResolveCache::new();
        cache.insert("a.rs".to_string(), "foo".to_string(), vec!["r".to_string()]);
        let _ = cache.get("a.rs", "foo");

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_default_capacity() {
        let cache = ResolveCache::new();
        assert_eq!(cache.stats().capacity, 100_000);
    }
}
