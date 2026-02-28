//! Project Resolution from File Paths
//!
//! Resolves a project_id from a file path or working directory by finding
//! the longest-prefix match against registered project `root_path`s.
//!
//! This module is used by:
//! - `SkillActivationHook` (in-process hook callback for PreToolUse)
//! - `GET /api/hooks/resolve-project` REST endpoint
//!
//! Results are cached for 5 minutes to avoid repeated Neo4j lookups.

use crate::neo4j::traits::GraphStore;
use crate::skills::hook_extractor::extract_file_context;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

// ============================================================================
// Types
// ============================================================================

/// Cache entry for project resolution: maps a project to its resolved info.
pub(crate) struct ResolvedProject {
    pub project_id: Uuid,
    pub slug: String,
    pub root_path: String,
}

/// A lightweight struct returned from find_longest_prefix_match
/// to avoid lifetime issues with the cache lock.
pub(crate) struct MatchedProject {
    pub project_id: Uuid,
    pub slug: String,
    pub root_path: String,
}

// ============================================================================
// Cache
// ============================================================================

/// TTL for the resolve-project cache (5 minutes).
pub(crate) const RESOLVE_CACHE_TTL: Duration = Duration::from_secs(300);

/// Global cache for project resolution.
/// Key: the full list of project entries (small — typically <20 projects).
/// Value: (entries, timestamp).
///
/// We cache the full project list (expanded root_paths) with a short TTL
/// rather than individual path lookups, because:
/// 1. Project count is small (typically <20)
/// 2. Avoids cache key explosion (infinite distinct file paths)
/// 3. A single cache entry covers all files under a root_path
type ResolveCache = Option<(Vec<ResolvedProject>, Instant)>;

pub(crate) static RESOLVE_CACHE: LazyLock<Mutex<ResolveCache>> = LazyLock::new(|| Mutex::new(None));

// ============================================================================
// Core Functions
// ============================================================================

/// Find the project whose root_path is the longest prefix of the given path.
///
/// Example: if projects have root_paths `/a/b/` and `/a/b/c/`,
/// and the input is `/a/b/c/d/file.rs`, the match is `/a/b/c/`.
pub(crate) fn find_longest_prefix_match(
    entries: &[ResolvedProject],
    path: &str,
) -> Option<MatchedProject> {
    // Normalize input: ensure it can be compared with trailing-slash root_paths
    // For a file path like /a/b/c/file.rs, we check if it starts with /a/b/c/
    // For a dir path like /a/b/c/, it naturally starts with /a/b/c/
    let check_path = if path.ends_with('/') {
        path.to_string()
    } else {
        format!("{}/", path)
    };

    let mut best: Option<&ResolvedProject> = None;
    let mut best_len = 0;

    for entry in entries {
        if check_path.starts_with(&entry.root_path) && entry.root_path.len() > best_len {
            best_len = entry.root_path.len();
            best = Some(entry);
        }
    }

    best.map(|b| MatchedProject {
        project_id: b.project_id,
        slug: b.slug.clone(),
        root_path: b.root_path.clone(),
    })
}

/// Load project entries from Neo4j (or cache), expanding root_paths.
///
/// Returns the cached entries if still valid, otherwise fetches from graph_store
/// and updates the cache.
pub(crate) async fn load_project_entries(
    graph_store: &dyn GraphStore,
) -> anyhow::Result<Vec<ResolvedProject>> {
    // Try the cache first
    {
        let cache = RESOLVE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((ref entries, ref cached_at)) = *cache {
            if cached_at.elapsed() < RESOLVE_CACHE_TTL {
                // Cache is valid — return a clone of the entries
                return Ok(entries
                    .iter()
                    .map(|e| ResolvedProject {
                        project_id: e.project_id,
                        slug: e.slug.clone(),
                        root_path: e.root_path.clone(),
                    })
                    .collect());
            }
        }
    }

    // Cache miss or expired — fetch all projects from Neo4j
    let projects = graph_store.list_projects().await?;

    let now = Instant::now();
    let entries: Vec<ResolvedProject> = projects
        .iter()
        .map(|p| {
            let expanded = crate::expand_tilde(&p.root_path);
            // Ensure root_path ends with / for correct prefix matching
            let normalized = if expanded.ends_with('/') {
                expanded
            } else {
                format!("{}/", expanded)
            };
            ResolvedProject {
                project_id: p.id,
                slug: p.slug.clone(),
                root_path: normalized,
            }
        })
        .collect();

    // Update cache
    {
        let mut cache = RESOLVE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        *cache = Some((
            entries
                .iter()
                .map(|e| ResolvedProject {
                    project_id: e.project_id,
                    slug: e.slug.clone(),
                    root_path: e.root_path.clone(),
                })
                .collect(),
            now,
        ));
    }

    Ok(entries)
}

/// Resolve a project_id from a tool call context.
///
/// Tries to find a matching project by:
/// 1. Extracting a file path from the tool_input (e.g., `file_path` from Read/Edit/Write)
/// 2. If found → longest-prefix match against project root_paths
/// 3. If not → fallback to longest-prefix match on the `cwd`
///
/// Returns `None` if no project matches either path.
///
/// # Arguments
///
/// * `graph_store` - Access to Neo4j for project list (cached for 5 min)
/// * `tool_name` - Claude Code tool name (e.g., "Read", "Bash", "Grep")
/// * `tool_input` - Raw JSON input of the tool call
/// * `cwd` - Working directory of the Claude Code session
pub async fn resolve_project_from_context(
    graph_store: &dyn GraphStore,
    tool_name: &str,
    tool_input: &serde_json::Value,
    cwd: &str,
) -> anyhow::Result<Option<Uuid>> {
    let entries = load_project_entries(graph_store).await?;

    // Try file path from tool_input first
    if let Some(file_path) = extract_file_context(tool_name, tool_input) {
        let normalized = crate::expand_tilde(&file_path);
        if let Some(matched) = find_longest_prefix_match(&entries, &normalized) {
            return Ok(Some(matched.project_id));
        }
    }

    // Fallback: try cwd
    let normalized_cwd = crate::expand_tilde(cwd);
    if let Some(matched) = find_longest_prefix_match(&entries, &normalized_cwd) {
        return Ok(Some(matched.project_id));
    }

    Ok(None)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, slug: &str, root: &str) -> ResolvedProject {
        ResolvedProject {
            project_id: Uuid::parse_str(id).unwrap(),
            slug: slug.to_string(),
            root_path: if root.ends_with('/') {
                root.to_string()
            } else {
                format!("{}/", root)
            },
        }
    }

    #[test]
    fn test_longest_prefix_match_single_project() {
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "my-project",
            "/Users/dev/projects/my-project",
        )];

        let result =
            find_longest_prefix_match(&entries, "/Users/dev/projects/my-project/src/main.rs");
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.slug, "my-project");
    }

    #[test]
    fn test_longest_prefix_match_no_match() {
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "my-project",
            "/Users/dev/projects/my-project",
        )];

        let result = find_longest_prefix_match(&entries, "/Users/dev/other-dir/something.rs");
        assert!(result.is_none());
    }

    #[test]
    fn test_longest_prefix_match_picks_longest() {
        let entries = vec![
            make_entry(
                "00000000-0000-0000-0000-000000000001",
                "workspace",
                "/Users/dev/workspace",
            ),
            make_entry(
                "00000000-0000-0000-0000-000000000002",
                "sub-project",
                "/Users/dev/workspace/packages/sub-project",
            ),
        ];

        // File in sub-project → should match sub-project (longer prefix), not workspace
        let result = find_longest_prefix_match(
            &entries,
            "/Users/dev/workspace/packages/sub-project/src/lib.rs",
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap().slug, "sub-project");

        // File in workspace root → should match workspace
        let result = find_longest_prefix_match(&entries, "/Users/dev/workspace/README.md");
        assert!(result.is_some());
        assert_eq!(result.unwrap().slug, "workspace");
    }

    #[test]
    fn test_longest_prefix_match_exact_root_path() {
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "my-project",
            "/Users/dev/my-project",
        )];

        // Passing the root_path itself (as directory) should match
        let result = find_longest_prefix_match(&entries, "/Users/dev/my-project");
        assert!(result.is_some());
        assert_eq!(result.unwrap().slug, "my-project");

        // With trailing slash
        let result = find_longest_prefix_match(&entries, "/Users/dev/my-project/");
        assert!(result.is_some());
        assert_eq!(result.unwrap().slug, "my-project");
    }

    #[test]
    fn test_longest_prefix_match_no_false_positive_on_similar_names() {
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "foo",
            "/Users/dev/foo",
        )];

        // /Users/dev/foobar should NOT match /Users/dev/foo/
        // because the trailing / in root_path prevents false positives
        let result = find_longest_prefix_match(&entries, "/Users/dev/foobar/src/main.rs");
        assert!(result.is_none());
    }

    #[test]
    fn test_longest_prefix_match_multiple_projects() {
        let entries = vec![
            make_entry(
                "00000000-0000-0000-0000-000000000001",
                "alpha",
                "/Users/dev/alpha",
            ),
            make_entry(
                "00000000-0000-0000-0000-000000000002",
                "beta",
                "/Users/dev/beta",
            ),
            make_entry(
                "00000000-0000-0000-0000-000000000003",
                "gamma",
                "/opt/projects/gamma",
            ),
        ];

        let r = find_longest_prefix_match(&entries, "/Users/dev/alpha/src/lib.rs");
        assert_eq!(r.unwrap().slug, "alpha");

        let r = find_longest_prefix_match(&entries, "/Users/dev/beta/tests/test.rs");
        assert_eq!(r.unwrap().slug, "beta");

        let r = find_longest_prefix_match(&entries, "/opt/projects/gamma/main.py");
        assert_eq!(r.unwrap().slug, "gamma");

        let r = find_longest_prefix_match(&entries, "/completely/different/path");
        assert!(r.is_none());
    }

    #[test]
    fn test_longest_prefix_match_empty_entries() {
        let entries: Vec<ResolvedProject> = vec![];
        let result = find_longest_prefix_match(&entries, "/some/path");
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_project_from_context_with_file_path() {
        // This is an async test that needs a mock GraphStore.
        // For now, test the synchronous find_longest_prefix_match directly.
        // Full async tests with mock GraphStore are in integration tests.
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "my-project",
            "/Users/dev/my-project",
        )];

        // Simulate extract_file_context("Read", {"file_path": "/Users/dev/my-project/src/main.rs"})
        let file_path = "/Users/dev/my-project/src/main.rs";
        let result = find_longest_prefix_match(&entries, file_path);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap().project_id,
            Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
        );
    }

    #[test]
    fn test_resolve_project_fallback_to_cwd() {
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "my-project",
            "/Users/dev/my-project",
        )];

        // No file_path extracted → fall back to cwd
        let cwd = "/Users/dev/my-project";
        let result = find_longest_prefix_match(&entries, cwd);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap().project_id,
            Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
        );
    }

    #[test]
    fn test_resolve_project_no_match() {
        let entries = vec![make_entry(
            "00000000-0000-0000-0000-000000000001",
            "my-project",
            "/Users/dev/my-project",
        )];

        // Neither file_path nor cwd match any project
        let result = find_longest_prefix_match(&entries, "/completely/different/path");
        assert!(result.is_none());
    }
}
