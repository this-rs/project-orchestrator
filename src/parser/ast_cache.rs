//! LRU cache for parsed AST results.
//!
//! Caches [`ParsedFile`] by `(path, content_hash)` to avoid re-parsing
//! unchanged files during incremental sync.
//!
//! # Design notes
//!
//! We cache `ParsedFile` (not `tree_sitter::Tree`) because:
//! - `Tree` is not `Send` — can't be shared across rayon threads
//! - `ParsedFile` includes all extracted symbols, skipping the extract step too
//! - `ParsedFile` is `Clone` and relatively lightweight
//!
//! The cache key is `"path:hash"` — same path with different content
//! gets a new entry, naturally evicting the old one via LRU.

use lru::LruCache;
use std::num::NonZeroUsize;

use super::ParsedFile;

/// Default maximum number of cached parse results.
///
/// Set to 50_000 to cover even the largest monorepos across multiple sync
/// cycles without eviction. Memory cost: ~600 MB at full capacity (~12 KB
/// per ParsedFile average), but in practice usage stays well below the cap
/// since entries are keyed by `path:content_hash` and LRU evicts stale ones.
pub const DEFAULT_AST_CACHE_CAPACITY: usize = 50_000;

/// Statistics about the AST cache.
#[derive(Debug, Clone, Copy)]
pub struct AstCacheStats {
    /// Current number of cached entries
    pub size: usize,
    /// Maximum capacity
    pub max_size: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
}

impl AstCacheStats {
    /// Cache hit rate as a percentage (0.0 – 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// LRU cache for parsed file results.
///
/// Thread-safety: this cache is NOT thread-safe. When used with rayon,
/// wrap it in `Arc<Mutex<AstCache>>` or access it only from the main thread.
pub struct AstCache {
    cache: LruCache<String, ParsedFile>,
    hits: u64,
    misses: u64,
}

impl AstCache {
    /// Create a new AST cache with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_AST_CACHE_CAPACITY)
    }

    /// Create a new AST cache with a specific capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(
                NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap()),
            ),
            hits: 0,
            misses: 0,
        }
    }

    /// Build the cache key from path and content hash.
    fn make_key(path: &str, content_hash: &str) -> String {
        format!("{}:{}", path, content_hash)
    }

    /// Look up a cached parse result.
    ///
    /// Returns `Some(ParsedFile)` on cache hit, `None` on miss.
    pub fn get(&mut self, path: &str, content_hash: &str) -> Option<&ParsedFile> {
        let key = Self::make_key(path, content_hash);
        if let Some(entry) = self.cache.get(&key) {
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store a parse result in the cache.
    ///
    /// If the cache is full, the least-recently-used entry is evicted.
    pub fn set(&mut self, path: &str, content_hash: &str, parsed: ParsedFile) {
        let key = Self::make_key(path, content_hash);
        self.cache.put(key, parsed);
    }

    /// Clear all cached entries. Hit/miss counters are preserved.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> AstCacheStats {
        AstCacheStats {
            size: self.cache.len(),
            max_size: self.cache.cap().get(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Reset hit/miss counters (useful between sync runs).
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

impl Default for AstCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parsed(path: &str, hash: &str) -> ParsedFile {
        ParsedFile {
            path: path.to_string(),
            language: "rust".to_string(),
            hash: hash.to_string(),
            functions: vec![],
            structs: vec![],
            traits: vec![],
            enums: vec![],
            imports: vec![],
            impl_blocks: vec![],
            function_calls: vec![],
            symbols: vec![],
        }
    }

    #[test]
    fn test_cache_hit_and_miss() {
        let mut cache = AstCache::new();

        // Miss
        assert!(cache.get("src/main.rs", "abc123").is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Set
        cache.set(
            "src/main.rs",
            "abc123",
            make_parsed("src/main.rs", "abc123"),
        );

        // Hit
        let result = cache.get("src/main.rs", "abc123");
        assert!(result.is_some());
        assert_eq!(result.unwrap().path, "src/main.rs");
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_cache_different_hash_is_miss() {
        let mut cache = AstCache::new();
        cache.set(
            "src/main.rs",
            "hash_v1",
            make_parsed("src/main.rs", "hash_v1"),
        );

        // Same path, different hash → miss
        assert!(cache.get("src/main.rs", "hash_v2").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = AstCache::with_capacity(2);

        cache.set("a.rs", "h1", make_parsed("a.rs", "h1"));
        cache.set("b.rs", "h2", make_parsed("b.rs", "h2"));
        assert_eq!(cache.stats().size, 2);

        // Adding a third evicts the LRU (a.rs)
        cache.set("c.rs", "h3", make_parsed("c.rs", "h3"));
        assert_eq!(cache.stats().size, 2);
        assert!(cache.get("a.rs", "h1").is_none()); // evicted
        assert!(cache.get("c.rs", "h3").is_some()); // still there
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = AstCache::new();
        cache.set("a.rs", "h1", make_parsed("a.rs", "h1"));
        cache.set("b.rs", "h2", make_parsed("b.rs", "h2"));
        assert_eq!(cache.stats().size, 2);

        cache.clear();
        assert_eq!(cache.stats().size, 0);
        assert!(cache.get("a.rs", "h1").is_none());

        // Counters preserved after clear
        assert!(cache.stats().misses > 0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut cache = AstCache::new();
        cache.set("a.rs", "h1", make_parsed("a.rs", "h1"));

        // 1 miss
        cache.get("b.rs", "h2");
        // 2 hits
        cache.get("a.rs", "h1");
        cache.get("a.rs", "h1");

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_reset_stats() {
        let mut cache = AstCache::new();
        cache.set("a.rs", "h1", make_parsed("a.rs", "h1"));
        cache.get("a.rs", "h1"); // hit
        cache.get("b.rs", "h2"); // miss

        cache.reset_stats();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.size, 1); // entries still there
    }

    #[test]
    fn test_cache_default_capacity() {
        let cache = AstCache::new();
        assert_eq!(cache.stats().max_size, DEFAULT_AST_CACHE_CAPACITY);
    }
}
