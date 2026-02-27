//! SuffixIndex: O(1) file lookup by path suffix
//!
//! For each file in the project, all possible path suffixes are indexed.
//! Example: `src/api/handlers.rs` generates these suffixes:
//! - `src/api/handlers.rs`
//! - `api/handlers.rs`
//! - `handlers.rs`
//!
//! This allows imports like `use crate::api::handlers` to resolve instantly
//! without scanning the entire file tree.
//!
//! Additionally, a directory index supports Java-style wildcard imports
//! (e.g., `import com.example.*`) by mapping `(dir_suffix, extension)` to files.

use std::collections::HashMap;

/// O(1) file lookup by path suffix.
///
/// Built once per sync, then used for all import resolutions in the project.
#[derive(Debug, Clone)]
pub struct SuffixIndex {
    /// suffix → full_path (first match wins for ambiguous suffixes)
    suffix_map: HashMap<String, SuffixEntry>,
    /// lowercase(suffix) → full_path for case-insensitive lookups
    lowercase_map: HashMap<String, SuffixEntry>,
    /// (dir_suffix, extension) → list of full paths
    dir_index: HashMap<(String, String), Vec<String>>,
    /// Total number of indexed files
    file_count: usize,
}

/// Entry for a suffix: either a unique match or ambiguous (multiple files share the suffix)
#[derive(Debug, Clone)]
enum SuffixEntry {
    /// Unique match — suffix resolves to exactly one file
    Unique(String),
    /// Ambiguous — multiple files share this suffix, store the first one found
    Ambiguous,
}

impl SuffixIndex {
    /// Build a SuffixIndex from a list of file paths.
    ///
    /// Complexity: O(files × average_path_depth)
    ///
    /// # Arguments
    /// * `paths` - Slice of file paths (relative to project root, using `/` separator)
    pub fn build(paths: &[String]) -> Self {
        let estimated_suffixes = paths.len() * 4; // average ~4 segments per path
        let mut suffix_map: HashMap<String, SuffixEntry> =
            HashMap::with_capacity(estimated_suffixes);
        let mut lowercase_map: HashMap<String, SuffixEntry> =
            HashMap::with_capacity(estimated_suffixes);
        let mut dir_index: HashMap<(String, String), Vec<String>> = HashMap::new();

        for path in paths {
            // Normalize separators
            let normalized = path.replace('\\', "/");

            // Generate all suffixes for this path
            let parts: Vec<&str> = normalized.split('/').collect();
            for start in 0..parts.len() {
                let suffix = parts[start..].join("/");
                let suffix_lower = suffix.to_lowercase();

                // Insert into suffix_map (detect ambiguity)
                match suffix_map.get(&suffix) {
                    None => {
                        suffix_map.insert(suffix.clone(), SuffixEntry::Unique(normalized.clone()));
                    }
                    Some(SuffixEntry::Unique(_)) => {
                        suffix_map.insert(suffix.clone(), SuffixEntry::Ambiguous);
                    }
                    Some(SuffixEntry::Ambiguous) => {
                        // Already ambiguous, nothing to do
                    }
                }

                // Insert into lowercase_map
                match lowercase_map.get(&suffix_lower) {
                    None => {
                        lowercase_map.insert(suffix_lower, SuffixEntry::Unique(normalized.clone()));
                    }
                    Some(SuffixEntry::Unique(_)) => {
                        lowercase_map.insert(suffix_lower, SuffixEntry::Ambiguous);
                    }
                    Some(SuffixEntry::Ambiguous) => {}
                }
            }

            // Build directory index
            // Extract directory and extension
            if let Some(last_slash) = normalized.rfind('/') {
                let dir = &normalized[..last_slash];
                if let Some(dot) = normalized.rfind('.') {
                    let ext = &normalized[dot + 1..];
                    // Index all directory suffixes
                    let dir_parts: Vec<&str> = dir.split('/').collect();
                    for start in 0..dir_parts.len() {
                        let dir_suffix = dir_parts[start..].join("/");
                        dir_index
                            .entry((dir_suffix, ext.to_string()))
                            .or_default()
                            .push(normalized.clone());
                    }
                }
            } else if let Some(dot) = normalized.rfind('.') {
                // Root-level file
                let ext = &normalized[dot + 1..];
                dir_index
                    .entry(("".to_string(), ext.to_string()))
                    .or_default()
                    .push(normalized.clone());
            }
        }

        Self {
            suffix_map,
            lowercase_map,
            dir_index,
            file_count: paths.len(),
        }
    }

    /// Look up a file by its path suffix (case-sensitive, O(1)).
    ///
    /// Returns `None` if the suffix is ambiguous (matches multiple files)
    /// or not found.
    ///
    /// # Examples
    /// ```
    /// # use project_orchestrator::resolver::SuffixIndex;
    /// let paths = vec!["src/api/handlers.rs".to_string()];
    /// let index = SuffixIndex::build(&paths);
    /// assert_eq!(index.get("handlers.rs"), Some("src/api/handlers.rs"));
    /// assert_eq!(index.get("api/handlers.rs"), Some("src/api/handlers.rs"));
    /// ```
    pub fn get(&self, suffix: &str) -> Option<&str> {
        match self.suffix_map.get(suffix) {
            Some(SuffixEntry::Unique(path)) => Some(path.as_str()),
            _ => None,
        }
    }

    /// Look up a file by its path suffix, case-insensitive (O(1)).
    ///
    /// Returns `None` if ambiguous or not found.
    pub fn get_insensitive(&self, suffix: &str) -> Option<&str> {
        let lower = suffix.to_lowercase();
        match self.lowercase_map.get(&lower) {
            Some(SuffixEntry::Unique(path)) => Some(path.as_str()),
            _ => None,
        }
    }

    /// Get all files in a directory matching the given extension.
    ///
    /// Supports Java-style wildcard imports like `import com.example.*`
    /// by looking up `("com/example", "java")`.
    ///
    /// # Arguments
    /// * `dir_suffix` - Directory path suffix (using `/` separator)
    /// * `extension` - File extension without the dot (e.g., "java", "rs")
    pub fn get_files_in_dir(&self, dir_suffix: &str, extension: &str) -> Vec<&str> {
        self.dir_index
            .get(&(dir_suffix.to_string(), extension.to_string()))
            .map(|paths| paths.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Check if a suffix is ambiguous (matches multiple files).
    pub fn is_ambiguous(&self, suffix: &str) -> bool {
        matches!(self.suffix_map.get(suffix), Some(SuffixEntry::Ambiguous))
    }

    /// Get the number of indexed files.
    pub fn file_count(&self) -> usize {
        self.file_count
    }

    /// Get the total number of suffix entries (for diagnostics).
    pub fn suffix_count(&self) -> usize {
        self.suffix_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_paths() -> Vec<String> {
        vec![
            "src/api/handlers.rs".to_string(),
            "src/api/routes.rs".to_string(),
            "src/neo4j/client.rs".to_string(),
            "src/parser/mod.rs".to_string(),
            "src/parser/languages/rust.rs".to_string(),
            "src/parser/languages/python.rs".to_string(),
            "src/lib.rs".to_string(),
            "Cargo.toml".to_string(),
        ]
    }

    #[test]
    fn test_exact_lookup() {
        let index = SuffixIndex::build(&sample_paths());

        // Full path lookup
        assert_eq!(
            index.get("src/api/handlers.rs"),
            Some("src/api/handlers.rs")
        );
        // Partial suffix
        assert_eq!(index.get("api/handlers.rs"), Some("src/api/handlers.rs"));
        // Just filename (unique)
        assert_eq!(index.get("handlers.rs"), Some("src/api/handlers.rs"));
        assert_eq!(index.get("client.rs"), Some("src/neo4j/client.rs"));
        assert_eq!(index.get("Cargo.toml"), Some("Cargo.toml"));
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let index = SuffixIndex::build(&sample_paths());

        assert_eq!(
            index.get_insensitive("HANDLERS.RS"),
            Some("src/api/handlers.rs")
        );
        assert_eq!(
            index.get_insensitive("Api/Handlers.rs"),
            Some("src/api/handlers.rs")
        );
        assert_eq!(index.get_insensitive("cargo.toml"), Some("Cargo.toml"));
    }

    #[test]
    fn test_ambiguous_suffix() {
        let index = SuffixIndex::build(&sample_paths());

        // "mod.rs" only appears once, should be unique
        assert_eq!(index.get("mod.rs"), Some("src/parser/mod.rs"));

        // But "rust.rs" and "python.rs" are unique filenames
        assert_eq!(index.get("rust.rs"), Some("src/parser/languages/rust.rs"));
        assert_eq!(
            index.get("python.rs"),
            Some("src/parser/languages/python.rs")
        );
    }

    #[test]
    fn test_truly_ambiguous_suffix() {
        // Two files with the same name in different directories
        let paths = vec![
            "src/api/mod.rs".to_string(),
            "src/parser/mod.rs".to_string(),
        ];
        let index = SuffixIndex::build(&paths);

        // "mod.rs" is ambiguous
        assert!(index.is_ambiguous("mod.rs"));
        assert_eq!(index.get("mod.rs"), None);

        // But qualified suffixes are unique
        assert_eq!(index.get("api/mod.rs"), Some("src/api/mod.rs"));
        assert_eq!(index.get("parser/mod.rs"), Some("src/parser/mod.rs"));
    }

    #[test]
    fn test_directory_listing() {
        let paths = vec![
            "src/com/example/User.java".to_string(),
            "src/com/example/Order.java".to_string(),
            "src/com/example/Product.java".to_string(),
            "src/com/other/Config.java".to_string(),
        ];
        let index = SuffixIndex::build(&paths);

        let files = index.get_files_in_dir("com/example", "java");
        assert_eq!(files.len(), 3);
        assert!(files.contains(&"src/com/example/User.java"));
        assert!(files.contains(&"src/com/example/Order.java"));
        assert!(files.contains(&"src/com/example/Product.java"));

        let other_files = index.get_files_in_dir("com/other", "java");
        assert_eq!(other_files.len(), 1);
        assert!(other_files.contains(&"src/com/other/Config.java"));

        // Non-existent directory
        let empty = index.get_files_in_dir("com/nonexistent", "java");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_empty_index() {
        let index = SuffixIndex::build(&[]);

        assert_eq!(index.get("anything.rs"), None);
        assert_eq!(index.get_insensitive("anything.rs"), None);
        assert!(index.get_files_in_dir("any", "rs").is_empty());
        assert_eq!(index.file_count(), 0);
    }

    #[test]
    fn test_special_characters_in_paths() {
        let paths = vec![
            "src/my-module/file_name.rs".to_string(),
            "src/my.module/other.file.rs".to_string(),
        ];
        let index = SuffixIndex::build(&paths);

        assert_eq!(
            index.get("my-module/file_name.rs"),
            Some("src/my-module/file_name.rs")
        );
        assert_eq!(
            index.get("my.module/other.file.rs"),
            Some("src/my.module/other.file.rs")
        );
    }

    #[test]
    fn test_windows_path_normalization() {
        let paths = vec!["src\\api\\handlers.rs".to_string()];
        let index = SuffixIndex::build(&paths);

        assert_eq!(index.get("api/handlers.rs"), Some("src/api/handlers.rs"));
        assert_eq!(index.get("handlers.rs"), Some("src/api/handlers.rs"));
    }

    #[test]
    fn test_diagnostics() {
        let index = SuffixIndex::build(&sample_paths());

        assert_eq!(index.file_count(), 8);
        assert!(
            index.suffix_count() > 8,
            "Should have more suffixes than files"
        );
    }

    #[test]
    fn test_large_index_performance() {
        // Generate 10K synthetic paths
        let paths: Vec<String> = (0..10_000)
            .map(|i| format!("src/module_{}/sub_{}/file_{}.rs", i / 100, i / 10 % 10, i))
            .collect();

        let start = std::time::Instant::now();
        let index = SuffixIndex::build(&paths);
        let build_time = start.elapsed();

        assert!(
            build_time.as_millis() < 500,
            "Build time should be < 500ms for 10K files, got {:?}",
            build_time
        );

        assert_eq!(index.file_count(), 10_000);

        // Lookup performance
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = index.get("sub_5/file_5050.rs");
        }
        let lookup_time = start.elapsed();

        assert!(
            lookup_time.as_micros() < 100_000,
            "10K lookups should be < 100ms, got {:?}",
            lookup_time
        );
    }
}
