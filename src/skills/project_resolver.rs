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
//!
//! Paths are compared in BOTH their given and canonical (symlink-resolved)
//! forms. Sync stores `File.path` canonicalized while `Project.root_path` is
//! stored as typed — e.g. the backend is registered through a symlink
//! (`~/projects/project-orchestrator/backend` → `~/.openclaw/.../project-
//! orchestrator`) while the CLI reports the real path as cwd. Plain string
//! prefix matching then never matched ("no project matched" for every tool
//! call in that project).

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

/// `path` with `~` expanded, relative paths joined onto `base` (when given),
/// plus its canonical form when it exists on disk and differs.
pub(crate) fn path_forms(path: &str, base: Option<&str>) -> Vec<String> {
    let expanded = crate::expand_tilde(path);
    let absolute = match base {
        Some(base) if !expanded.starts_with('/') && !expanded.is_empty() => {
            let base = crate::expand_tilde(base);
            format!("{}/{}", base.trim_end_matches('/'), expanded)
        }
        _ => expanded,
    };
    let mut forms = vec![absolute.clone()];
    if let Ok(canonical) = std::fs::canonicalize(&absolute) {
        let canonical = canonical.display().to_string();
        if canonical != absolute {
            forms.push(canonical);
        }
    }
    forms
}

/// The form of `path` stored in the graph: `File.path` is canonical (sync
/// canonicalizes), so persona KNOWS lookups must use the canonical form.
/// Relative paths are joined onto `base`; falls back to the absolute form
/// when the file does not exist.
pub fn graph_file_path(path: &str, base: Option<&str>) -> String {
    path_forms(path, base)
        .pop()
        .unwrap_or_else(|| path.to_string())
}

/// Longest-prefix match over every form of `path` (see [`path_forms`]).
pub(crate) fn find_match_any_form(
    entries: &[ResolvedProject],
    forms: &[String],
) -> Option<MatchedProject> {
    forms
        .iter()
        .filter_map(|form| find_longest_prefix_match(entries, form))
        .max_by_key(|m| m.root_path.len())
}

/// Resolver entries for `projects`: one per root_path form (as stored, and
/// canonical when it differs), each ending with `/`.
pub(crate) fn entries_for_projects(
    projects: &[crate::neo4j::models::ProjectNode],
) -> Vec<ResolvedProject> {
    let mut entries = Vec::new();
    for p in projects {
        if p.root_path.is_empty() {
            continue;
        }
        for form in path_forms(&p.root_path, None) {
            let root_path = if form.ends_with('/') {
                form
            } else {
                format!("{}/", form)
            };
            entries.push(ResolvedProject {
                project_id: p.id,
                slug: p.slug.clone(),
                root_path,
            });
        }
    }
    entries
}

/// Project owning `cwd`, for sessions created without a project_slug
/// (workspace / all-projects mode). Returns `None` when no project matches,
/// or when the best match is ambiguous (several projects registered on the
/// same root) — attaching a session to the wrong project is worse than none.
pub(crate) fn infer_project_slug(entries: &[ResolvedProject], cwd: &str) -> Option<String> {
    let best = find_match_any_form(entries, &path_forms(cwd, None))?;
    let tied: std::collections::HashSet<&str> = entries
        .iter()
        .filter(|e| e.root_path == best.root_path)
        .map(|e| e.slug.as_str())
        .collect();
    (tied.len() == 1).then_some(best.slug)
}

/// Async wrapper of [`infer_project_slug`] over the cached project list.
pub async fn infer_project_slug_for_cwd(graph_store: &dyn GraphStore, cwd: &str) -> Option<String> {
    if cwd.is_empty() {
        return None;
    }
    let entries = load_project_entries(graph_store).await.ok()?;
    infer_project_slug(&entries, cwd)
}

/// The forms of a tool's file path to match against a project's skill
/// triggers: relative to each of the project's roots first (file-glob
/// triggers are root-relative, e.g. `src/neo4j/**`), then the absolute
/// forms (legacy absolute triggers). Deduplicated, in that order.
pub(crate) fn file_candidates(
    entries: &[ResolvedProject],
    project_id: Uuid,
    forms: &[String],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let roots = entries.iter().filter(|e| e.project_id == project_id);
    for root in roots {
        for form in forms {
            if let Some(rel) = form.strip_prefix(root.root_path.as_str()) {
                if !rel.is_empty() && !out.iter().any(|o| o == rel) {
                    out.push(rel.to_string());
                }
            }
        }
    }
    for form in forms {
        if !out.iter().any(|o| o == form) {
            out.push(form.clone());
        }
    }
    out
}

/// Async wrapper of [`file_candidates`] over the cached project list.
/// `cwd` resolves relative tool paths.
pub async fn hook_file_candidates(
    graph_store: &dyn GraphStore,
    project_id: Uuid,
    file: &str,
    cwd: Option<&str>,
) -> Vec<String> {
    let forms = path_forms(file, cwd);
    let mut entries = match load_project_entries(graph_store).await {
        Ok(entries) => entries,
        Err(_) => return forms,
    };
    // The cached list can predate this project (5 min TTL): reload rather
    // than silently match without its root.
    if !entries.iter().any(|e| e.project_id == project_id) {
        if let Ok(projects) = graph_store.list_projects().await {
            entries = entries_for_projects(&projects);
        }
    }
    file_candidates(&entries, project_id, &forms)
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
    let entries = entries_for_projects(&projects);

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
    // Relative tool paths (Bash `cat src/a.rs`) are resolved against cwd.
    if let Some(file_path) = extract_file_context(tool_name, tool_input) {
        let forms = path_forms(&file_path, Some(cwd));
        if let Some(matched) = find_match_any_form(&entries, &forms) {
            return Ok(Some(matched.project_id));
        }
    }

    // Fallback: try cwd
    if let Some(matched) = find_match_any_form(&entries, &path_forms(cwd, None)) {
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

    /// A project registered through a symlink, like the prod backend.
    fn symlinked_project() -> (tempfile::TempDir, String, String) {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real-project");
        std::fs::create_dir_all(real.join("src")).unwrap();
        std::fs::write(real.join("src/main.rs"), "").unwrap();
        let link = tmp.path().join("link-project");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let real = std::fs::canonicalize(&real).unwrap().display().to_string();
        (tmp, link.display().to_string(), real)
    }

    fn project_node(slug: &str, root: &str) -> crate::neo4j::models::ProjectNode {
        let mut project = crate::test_helpers::test_project_named(slug);
        project.slug = slug.to_string();
        project.root_path = root.to_string();
        project
    }

    #[test]
    fn test_symlinked_root_matches_the_real_path() {
        // Regression: File.path and the CLI cwd are canonical, root_path is
        // the symlink — "no project matched" for every tool call.
        let (_tmp, link, real) = symlinked_project();
        let entries = entries_for_projects(&[project_node("backend", &link)]);

        let file = format!("{real}/src/main.rs");
        let matched = find_match_any_form(&entries, &path_forms(&file, None));
        assert_eq!(matched.map(|m| m.slug), Some("backend".to_string()));
        // And the symlinked form still matches too.
        let via_link = format!("{link}/src/main.rs");
        assert!(find_match_any_form(&entries, &path_forms(&via_link, None)).is_some());
    }

    #[test]
    fn test_relative_tool_path_is_resolved_against_cwd() {
        let (_tmp, link, real) = symlinked_project();
        let entries = entries_for_projects(&[project_node("backend", &link)]);
        let forms = path_forms("src/main.rs", Some(&real));
        assert_eq!(forms[0], format!("{real}/src/main.rs"));
        assert!(find_match_any_form(&entries, &forms).is_some());
    }

    #[test]
    fn test_infer_project_slug_from_cwd() {
        // Sessions created in all-projects mode carry no project_slug; the
        // cwd (the selected project's root) identifies the project.
        let (_tmp, link, real) = symlinked_project();
        let entries = entries_for_projects(&[project_node("backend", &link)]);
        assert_eq!(
            infer_project_slug(&entries, &real),
            Some("backend".to_string())
        );
        assert_eq!(
            infer_project_slug(&entries, &link),
            Some("backend".to_string())
        );
        assert_eq!(infer_project_slug(&entries, "/somewhere/else"), None);
    }

    #[test]
    fn test_infer_project_slug_refuses_ambiguous_roots() {
        // Two projects registered on the same root (happens in prod: obrain
        // and grafeo) — attaching the session to either would be a guess.
        let entries = entries_for_projects(&[
            project_node("obrain", "/Users/dev/lab/grafeo"),
            project_node("grafeo", "/Users/dev/lab/grafeo"),
            project_node("other", "/Users/dev/lab/other"),
        ]);
        assert_eq!(infer_project_slug(&entries, "/Users/dev/lab/grafeo"), None);
        assert_eq!(
            infer_project_slug(&entries, "/Users/dev/lab/other/src"),
            Some("other".to_string())
        );
    }

    #[test]
    fn test_graph_file_path_is_canonical() {
        let (_tmp, link, real) = symlinked_project();
        assert_eq!(
            graph_file_path(&format!("{link}/src/main.rs"), None),
            format!("{real}/src/main.rs")
        );
        assert_eq!(
            graph_file_path("src/main.rs", Some(&link)),
            format!("{real}/src/main.rs")
        );
        // Missing files keep their absolute form.
        assert_eq!(graph_file_path("/nope/x.rs", None), "/nope/x.rs");
    }

    #[test]
    fn test_file_candidates_are_root_relative_first() {
        // File-glob triggers are root-relative (`src/**`): a Read of an
        // absolute path — canonical, under a symlinked root — must be offered
        // to them as `src/main.rs`.
        let (_tmp, link, real) = symlinked_project();
        let project = project_node("backend", &link);
        let entries = entries_for_projects(std::slice::from_ref(&project));
        let file = format!("{real}/src/main.rs");
        let candidates = file_candidates(&entries, project.id, &path_forms(&file, None));
        assert_eq!(candidates[0], "src/main.rs");
        assert!(
            candidates.contains(&file),
            "absolute form kept for legacy triggers"
        );
        // Another project's roots are not used.
        let other = file_candidates(&entries, Uuid::new_v4(), &path_forms(&file, None));
        assert!(!other.contains(&"src/main.rs".to_string()));
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
