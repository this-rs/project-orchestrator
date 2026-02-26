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
    /// PHP PSR-4 mappings from composer.json (namespace prefix → directory), lazy-loaded
    pub psr4_mappings: std::collections::HashMap<String, String>,
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
            psr4_mappings: std::collections::HashMap::new(),
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

    /// Load PHP PSR-4 autoload mappings from composer.json.
    ///
    /// Parses `composer.json` in the project root and extracts
    /// `autoload.psr-4` namespace-to-directory mappings.
    pub fn load_composer_psr4(&mut self, project_root: &str) {
        if !self.psr4_mappings.is_empty() {
            return; // Already loaded
        }
        self.psr4_mappings = parse_composer_psr4(project_root);
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

/// Parse `composer.json` to extract PSR-4 autoload mappings.
///
/// Looks for `autoload.psr-4` and returns a map of namespace prefix → directory.
/// Namespace prefixes are normalized: trailing `\\` is preserved for matching.
/// Returns an empty map if composer.json doesn't exist or has no PSR-4 config.
pub fn parse_composer_psr4(project_root: &str) -> std::collections::HashMap<String, String> {
    let composer_path = std::path::Path::new(project_root).join("composer.json");
    let content = match std::fs::read_to_string(&composer_path) {
        Ok(c) => c,
        Err(_) => return std::collections::HashMap::new(),
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return std::collections::HashMap::new(),
    };

    let mut mappings = std::collections::HashMap::new();

    if let Some(psr4) = json
        .get("autoload")
        .and_then(|a| a.get("psr-4"))
        .and_then(|p| p.as_object())
    {
        for (namespace, dir_val) in psr4 {
            if let Some(dir) = dir_val.as_str() {
                mappings.insert(namespace.clone(), dir.to_string());
            }
        }
    }

    // Also check autoload-dev for test namespaces
    if let Some(psr4) = json
        .get("autoload-dev")
        .and_then(|a| a.get("psr-4"))
        .and_then(|p| p.as_object())
    {
        for (namespace, dir_val) in psr4 {
            if let Some(dir) = dir_val.as_str() {
                mappings.insert(namespace.clone(), dir.to_string());
            }
        }
    }

    mappings
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

    #[test]
    fn test_parse_composer_psr4_basic() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("composer.json"),
            r#"{"autoload":{"psr-4":{"App\\":"src/","Domain\\":"lib/domain/"}}}"#,
        )
        .unwrap();

        let result = parse_composer_psr4(dir.path().to_str().unwrap());
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("App\\").unwrap(), "src/");
        assert_eq!(result.get("Domain\\").unwrap(), "lib/domain/");
    }

    #[test]
    fn test_parse_composer_psr4_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = parse_composer_psr4(dir.path().to_str().unwrap());
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_composer_psr4_no_autoload() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("composer.json"),
            r#"{"name":"vendor/pkg","require":{"php":">=8.1"}}"#,
        )
        .unwrap();

        let result = parse_composer_psr4(dir.path().to_str().unwrap());
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_composer_psr4_with_dev() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("composer.json"),
            r#"{
                "autoload": {"psr-4": {"App\\": "src/"}},
                "autoload-dev": {"psr-4": {"Tests\\": "tests/"}}
            }"#,
        )
        .unwrap();

        let result = parse_composer_psr4(dir.path().to_str().unwrap());
        assert_eq!(result.len(), 2);
        assert_eq!(result.get("App\\").unwrap(), "src/");
        assert_eq!(result.get("Tests\\").unwrap(), "tests/");
    }
}
