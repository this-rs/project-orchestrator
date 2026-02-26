//! Import resolution infrastructure
//!
//! Provides data structures and algorithms for resolving import paths
//! to actual files in the project:
//! - `SuffixIndex`: O(1) file lookup by path suffix
//! - `SymbolTable`: dual-index symbol resolution (by name and by file)
//! - `ResolveCache`: LRU cache for resolved imports

pub mod resolve_cache;
pub mod suffix_index;
pub mod symbol_table;

pub use resolve_cache::ResolveCache;
pub use suffix_index::SuffixIndex;
pub use symbol_table::SymbolTable;

/// Context for import resolution during a sync pass.
///
/// Built once after collecting all file paths, then shared across
/// all per-file import resolution calls.
#[derive(Debug)]
pub struct ImportResolutionContext {
    /// O(1) file lookup by path suffix
    pub suffix_index: SuffixIndex,
    /// Dual-index symbol resolution
    pub symbol_table: SymbolTable,
    /// LRU cache for resolved imports
    pub resolve_cache: ResolveCache,
}

impl ImportResolutionContext {
    /// Build a new context from a list of file paths.
    ///
    /// The symbol table starts empty and should be populated
    /// via `populate_symbols()` after all files are parsed.
    pub fn new(file_paths: &[String]) -> Self {
        Self {
            suffix_index: SuffixIndex::build(file_paths),
            symbol_table: SymbolTable::new(),
            resolve_cache: ResolveCache::new(),
        }
    }

    /// Populate the symbol table from parsed files.
    pub fn populate_symbols(&mut self, parsed_files: &[crate::parser::ParsedFile]) {
        self.symbol_table.populate_from_parsed(parsed_files);
    }

    /// Log diagnostics about the resolution context.
    pub fn log_stats(&self) {
        let sym_stats = self.symbol_table.stats();
        let cache_stats = self.resolve_cache.stats();
        tracing::info!(
            files = self.suffix_index.file_count(),
            suffixes = self.suffix_index.suffix_count(),
            symbols = sym_stats.total_definitions,
            cache_size = cache_stats.size,
            cache_hit_rate = format!("{:.1}%", cache_stats.hit_rate * 100.0),
            "Import resolution context stats"
        );
    }
}
