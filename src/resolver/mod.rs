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
    /// Go module path from go.mod (e.g. "github.com/user/project"), lazy-loaded
    pub go_module_path: Option<String>,
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
            go_module_path: None,
        }
    }

    /// Load Go module path from go.mod file.
    ///
    /// Searches for go.mod in the project root (inferred from file paths)
    /// and extracts the `module` directive.
    pub fn load_go_module_path(&mut self, project_root: &str) {
        if self.go_module_path.is_some() {
            return; // Already loaded
        }
        self.go_module_path = parse_go_mod_module(project_root);
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

/// Parse `go.mod` to extract the module path.
///
/// Scans the file for a line matching `module <path>` and returns it.
/// Returns `None` if go.mod doesn't exist or the module line is missing.
pub fn parse_go_mod_module(project_root: &str) -> Option<String> {
    let go_mod_path = std::path::Path::new(project_root).join("go.mod");
    let content = std::fs::read_to_string(&go_mod_path).ok()?;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("module ") {
            let module_path = trimmed
                .strip_prefix("module ")?
                .trim()
                .trim_end_matches(';') // rare but defensive
                .to_string();
            if !module_path.is_empty() {
                return Some(module_path);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_go_mod_module_basic() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("go.mod"),
            "module github.com/user/myproject\n\ngo 1.21\n",
        )
        .unwrap();

        let result = parse_go_mod_module(dir.path().to_str().unwrap());
        assert_eq!(result, Some("github.com/user/myproject".to_string()));
    }

    #[test]
    fn test_parse_go_mod_module_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = parse_go_mod_module(dir.path().to_str().unwrap());
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_go_mod_module_no_module_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "go 1.21\n").unwrap();

        let result = parse_go_mod_module(dir.path().to_str().unwrap());
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_go_mod_module_with_comments() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("go.mod"),
            "// This is a Go module\nmodule example.com/foo/bar\n\ngo 1.22\n",
        )
        .unwrap();

        let result = parse_go_mod_module(dir.path().to_str().unwrap());
        assert_eq!(result, Some("example.com/foo/bar".to_string()));
    }
}
