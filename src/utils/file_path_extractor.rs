//! Shared file path extraction from text content.
//!
//! Provides a single, robust implementation for extracting file paths mentioned
//! in free-form text (chat messages, notes, etc.). Consolidates 6 previously
//! divergent implementations into one canonical function.

use std::collections::HashSet;

/// Known source file extensions for path detection (without the dot prefix).
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs",
    "ts",
    "tsx",
    "js",
    "jsx",
    "py",
    "go",
    "java",
    "kt",
    "swift",
    "rb",
    "php",
    "cs",
    "cpp",
    "c",
    "h",
    "hpp",
    "toml",
    "yaml",
    "yml",
    "json",
    "sql",
    "sh",
    "bash",
    "zsh",
    "md",
    "html",
    "css",
    "scss",
    "vue",
    "svelte",
    "ex",
    "exs",
    "zig",
    "lua",
    "r",
    "m",
    "mm",
    "xml",
    "txt",
    "graphql",
    "proto",
    "dockerfile",
];

/// Known config/root file names that don't require a `/` separator.
const ROOT_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "go.sum",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".env",
    ".env.example",
    "README.md",
];

/// Extract file paths mentioned in text content.
///
/// Detects:
/// - Relative paths: `src/neo4j/client.rs`, `tests/unit/test_foo.py`
/// - Absolute paths: `/Users/foo/project/src/bar.rs` → normalized to relative after `src/`
/// - Backtick-wrapped: `` `src/neo4j/client.rs` ``
/// - Known root files without slash: `Cargo.toml`, `Makefile`
/// - Bare filenames with known extensions: `main.rs`, `config.yaml`
///
/// Filters out:
/// - URLs (`http://`, `https://`, `ftp://`)
/// - Rust module paths (`crate::foo::bar`)
/// - Invalid/too-short tokens
///
/// Returns deduplicated, alphabetically sorted paths.
pub fn extract_file_paths(text: &str) -> Vec<String> {
    let mut paths = HashSet::new();

    for line in text.lines() {
        let tokens = extract_path_tokens(line);
        for token in tokens {
            if let Some(path) = validate_and_normalize_path(&token) {
                paths.insert(path);
            }
        }
    }

    let mut result: Vec<String> = paths.into_iter().collect();
    result.sort();
    result
}

/// Extract potential path tokens from a line of text.
///
/// Handles backtick-wrapped paths and whitespace/punctuation-delimited tokens.
fn extract_path_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    // First: extract backtick-wrapped content (highest priority)
    let mut rest = line;
    while let Some(start) = rest.find('`') {
        let after_tick = &rest[start + 1..];
        if let Some(end) = after_tick.find('`') {
            let inside = &after_tick[..end];
            if !inside.contains(' ') && !inside.is_empty() {
                tokens.push(inside.to_string());
            }
            rest = &after_tick[end + 1..];
        } else {
            break;
        }
    }

    // Second: split line by whitespace and common delimiters
    for word in line.split(|c: char| {
        c.is_whitespace()
            || c == ','
            || c == ';'
            || c == '('
            || c == ')'
            || c == '['
            || c == ']'
            || c == '"'
            || c == '\''
    }) {
        let clean = word.trim_matches(|c: char| {
            c == '`' || c == '\'' || c == '"' || c == '(' || c == ')' || c == ':'
        });
        if !clean.is_empty() {
            tokens.push(clean.to_string());
        }
    }

    tokens
}

/// Validate a token as a file path and normalize it.
///
/// Returns `Some(normalized_path)` if the token looks like a valid source file path.
fn validate_and_normalize_path(token: &str) -> Option<String> {
    let token = token
        .trim()
        .trim_end_matches(['.', ',', ';', ':', ')', ']']);

    if token.is_empty() || token.len() < 3 || token.len() > 256 {
        return None;
    }

    // Reject URLs
    if token.starts_with("http://")
        || token.starts_with("https://")
        || token.starts_with("ftp://")
        || token.starts_with("//")
    {
        return None;
    }

    // Reject Rust module paths (::)
    if token.contains("::") {
        return None;
    }

    // Check for known root files (exact match, no slash needed)
    if ROOT_FILES.contains(&token) {
        return Some(token.to_string());
    }

    let has_slash = token.contains('/');

    // Check for known extension
    let has_known_extension = if let Some(dot_pos) = token.rfind('.') {
        let ext = &token[dot_pos + 1..];
        SOURCE_EXTENSIONS.contains(&ext.to_lowercase().as_str())
    } else {
        false
    };

    // Must have a slash or a known extension to qualify
    if !has_slash && !has_known_extension {
        return None;
    }

    // If no slash but has extension, accept as bare filename
    if !has_slash && has_known_extension {
        // Additional check: must have path-like characters only
        if token
            .chars()
            .all(|c| c.is_alphanumeric() || "._-+@".contains(c))
        {
            let normalized = token.strip_prefix("./").unwrap_or(token);
            return Some(normalized.to_string());
        }
        return None;
    }

    // Has a slash — validate as a path
    // Must have a recognized extension (for paths with slashes)
    if !has_known_extension {
        return None;
    }

    // Normalize: strip leading ./
    let token = token.strip_prefix("./").unwrap_or(token);

    // Normalize absolute paths: find "src/" or "tests/" and take from there
    let normalized = if token.starts_with('/') {
        if let Some(idx) = token.find("/src/") {
            &token[idx + 1..]
        } else if let Some(idx) = token.find("/tests/") {
            &token[idx + 1..]
        } else {
            return None;
        }
    } else {
        token
    };

    // Sanity: must have path-like characters only
    if !normalized
        .chars()
        .all(|c| c.is_alphanumeric() || "/_.-+@".contains(c))
    {
        return None;
    }

    // Must have at least dir/file structure for slash-paths
    let segments: Vec<&str> = normalized.split('/').collect();
    if segments.len() < 2 {
        return None;
    }

    // Reject if any segment is empty
    if segments.iter().any(|s| s.is_empty()) {
        return None;
    }

    Some(normalized.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_with_slash() {
        let paths = extract_file_paths("Look at src/chat/manager.rs and src/lib.rs");
        assert!(paths.contains(&"src/chat/manager.rs".to_string()));
        assert!(paths.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn test_path_without_slash_root_file() {
        let paths = extract_file_paths("Edit the Cargo.toml file");
        assert!(paths.contains(&"Cargo.toml".to_string()));
    }

    #[test]
    fn test_path_without_slash_bare_extension() {
        let paths = extract_file_paths("Check main.rs for the issue");
        assert!(paths.contains(&"main.rs".to_string()));
    }

    #[test]
    fn test_backtick_quoted_path() {
        let paths = extract_file_paths("Fix the bug in `src/neo4j/client.rs`");
        assert!(paths.contains(&"src/neo4j/client.rs".to_string()));
    }

    #[test]
    fn test_url_rejected() {
        let paths = extract_file_paths("See https://example.com/foo/bar.rs for details");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_rust_module_rejected() {
        let paths = extract_file_paths("Use crate::neo4j::client for the query");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_duplicates_deduplicated() {
        let paths = extract_file_paths("src/lib.rs and src/lib.rs again and `src/lib.rs`");
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "src/lib.rs");
    }

    #[test]
    fn test_sorted_output() {
        let paths = extract_file_paths("src/z.rs src/a.rs src/m.rs");
        assert_eq!(paths, vec!["src/a.rs", "src/m.rs", "src/z.rs"]);
    }

    #[test]
    fn test_no_paths() {
        let paths = extract_file_paths("Hello, how are you?");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_http_rejected() {
        let paths = extract_file_paths("http://localhost:8080/api/foo.json");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_absolute_path_normalized() {
        let paths = extract_file_paths("/Users/foo/project/src/main.rs");
        assert!(paths.contains(&"src/main.rs".to_string()));
    }

    #[test]
    fn test_dotslash_stripped() {
        let paths = extract_file_paths("./src/main.rs");
        assert!(paths.contains(&"src/main.rs".to_string()));
    }

    #[test]
    fn test_quoted_paths() {
        let paths = extract_file_paths(r#"Open "src/lib.rs" and 'tests/api_tests.rs'"#);
        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(paths.contains(&"tests/api_tests.rs".to_string()));
    }

    #[test]
    fn test_makefile() {
        let paths = extract_file_paths("Check the Makefile");
        assert!(paths.contains(&"Makefile".to_string()));
    }

    #[test]
    fn test_ftp_rejected() {
        let paths = extract_file_paths("ftp://server.com/path/file.rs");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_double_slash_rejected() {
        let paths = extract_file_paths("//comment/style/path.rs");
        assert!(paths.is_empty());
    }
}
