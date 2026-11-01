//! Stable identity of a *file*, independent of where it is mounted — and the
//! ONE rule for the form a path takes in the graph.
//!
//! A file used to be identified by its absolute path (`MERGE (f:File {path})`).
//! That made the same file a different node as soon as it was reached through
//! a symlink, a git worktree or a moved repository root — measured: the PO
//! backend was indexed under `~/.openclaw/workspace/skills/project-orchestrator`
//! while its declared `root_path` is `~/projects/project-orchestrator/backend`,
//! a symlink to the very same directory (same inode).
//!
//! [`file_uid`] derives the identity from the *project* and the path **relative
//! to the project root**, so the mounting point is no longer part of it. It is
//! computed HERE, in Rust, by the write paths (`upsert_file`,
//! `batch_upsert_files`) and by the uid backfill alike — one grammar, no Cypher
//! re-implementation to keep in step.
//!
//! [`PathPolicy`] is the stored form of a path for one project: canonical for a
//! project that has not been migrated yet, declared once it has
//! (`Project.paths_declared`). Every site that writes OR reads a `File.path`
//! goes through it — the scan, the single-file sync, the watcher's deletions
//! and invalidations, the persona lookups — so no site can change a project's
//! vocabulary on its own.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use uuid::Uuid;

/// Compute the stable uid of a file: `<project>:<relative_path>`.
///
/// `relative_path` MUST be relative to the project root and use `/` as
/// separator, so that the same file reached through another mount produces
/// the same uid. When there is no project (an ad-hoc sync), the absolute
/// path is still used — identity stays stable for that call site — and it is
/// kept **as given**, leading `/` included: the orphan backfill writes
/// `'-:' + f.path`, and one grammar on two sites means the same bytes on both.
///
/// # Why a readable form and not a hash
///
/// A concatenation is verifiable by eye: when a duplicate shows up, the reader
/// can see which part differs (project, or relative path). A digest would hide
/// exactly that.
///
/// The owner is the hyphenated project id, or `-` when there is none.
pub fn file_uid(project_id: Option<Uuid>, relative_path: &str) -> String {
    let owner = project_id
        .map(|id| id.hyphenated().to_string())
        .unwrap_or_else(|| "-".to_string());
    format!("{}:{}", owner, relative_path)
}

/// The declared root in the form every stored path of a declared project
/// starts with: `~` expanded, no trailing `/`. This is what `Project.declared_path`
/// holds, written in Rust — the Cypher side never derives anything from the raw
/// `root_path` (a tilde or a trailing slash there made `STARTS WITH` false and
/// the uid mount-dependent).
pub fn declared_root_of(root_path: &str) -> String {
    let expanded = crate::expand_tilde(root_path.trim());
    let trimmed = expanded.trim_end_matches('/');
    if trimmed.is_empty() && !expanded.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Resolve a declared root to its canonical form, or `None` when the directory
/// cannot be resolved (missing, unmounted, not yet created).
///
/// `~` is expanded first: roots may be stored with a tilde, and
/// `canonicalize("~/x")` fails on the literal tilde — which used to leave
/// `canonical_path = "~/…"`, a prefix no stored path ever matches.
pub fn try_canonical_root(declared: Option<&str>) -> Option<String> {
    let declared = declared.map(str::trim).filter(|r| !r.is_empty())?;
    std::fs::canonicalize(crate::expand_tilde(declared))
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

/// Canonical form of a project root: the path the graph stored before the
/// migration (the sync used to resolve links before writing).
///
/// Falls back to the normalized declared path when canonicalisation fails: a
/// project we cannot resolve is still a project, and an empty value would
/// silently drop it from the backfill. Callers that must not overwrite a good
/// value with this fallback use [`try_canonical_root`].
pub fn canonical_root(declared: Option<&str>) -> String {
    let Some(declared) = declared.map(str::trim).filter(|r| !r.is_empty()) else {
        return String::new();
    };
    try_canonical_root(Some(declared)).unwrap_or_else(|| declared_root_of(declared))
}

/// What the boot repair may write for one project's `canonical_path`.
///
/// `resolved` is [`try_canonical_root`] at THIS boot. A failed resolution (a
/// volume not mounted today) never overwrites a stored value: the stored one
/// was computed when the directory was there, and replacing it with the
/// declared fallback would silently change how every uid of the project is
/// derived. The fallback is written only when nothing is stored yet.
pub fn canonical_path_update(
    stored: &str,
    resolved: Option<&str>,
    declared: &str,
) -> Option<String> {
    match resolved {
        Some(r) if r != stored => Some(r.to_string()),
        Some(_) => None,
        None if stored.is_empty() => {
            let fallback = declared_root_of(declared);
            (!fallback.is_empty()).then_some(fallback)
        }
        None => None,
    }
}

/// `rest` of `path` after `root`, only at a `/` boundary.
///
/// `/data/proj-old/x.rs` is NOT under `/data/proj`: without the boundary the
/// strip would yield `-old/x.rs` and a relative path that names another
/// project's file.
fn strip_root<'a>(root: &str, path: &'a str) -> Option<&'a str> {
    let root = root.trim_end_matches('/');
    if root.is_empty() {
        return None;
    }
    let rest = path.strip_prefix(root)?;
    if rest.starts_with('/') && rest.len() > 1 {
        Some(rest)
    } else {
        None
    }
}

/// Strip `root` from `path`, both expected in the same form, at a `/` boundary.
///
/// Returns `None` when `path` is not under `root` — callers then keep the
/// absolute path rather than inventing a misleading relative one.
pub fn relative_to_root(root: &str, path: &str) -> Option<String> {
    strip_root(root, path).map(|rest| rest.trim_start_matches('/').to_string())
}

fn canonicalize_or_keep(path: &str) -> String {
    std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string())
}

/// The form a project's paths take in the graph, and how to bring any path to it.
///
/// Two forms exist for a project declared through a link: the **canonical**
/// one (what the sync used to write) and the **declared** one (what the user
/// typed, normalized by [`declared_root_of`]). A project's nodes are all in one
/// form; which one is recorded on the project (`paths_declared`), set by
/// `migrate file-paths --apply` once every rename pass has succeeded, and at
/// creation for a new project. Until then the sync keeps writing the canonical
/// form: changing the vocabulary before the nodes are renamed would create a
/// declared twin for every symbol the sync touches (symbols are merged on an
/// `id` that embeds the path), and a stale-file pass would then delete the
/// originals with their knowledge edges.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PathPolicy {
    /// `Project.declared_path`: tilde expanded, no trailing `/`. Empty: no project.
    pub declared_root: String,
    /// The canonical form of that root. Empty: unknown.
    pub canonical_root: String,
    /// `Project.paths_declared`: the stored form is the declared one.
    pub declared: bool,
}

impl PathPolicy {
    /// No project: a path is stored in its canonical form — the invariant the
    /// project-less syncs always had (one file, one node, whatever the mount).
    pub fn none() -> Self {
        Self::default()
    }

    /// Policy of a project from its roots. `canonical` may be the stored
    /// `Project.canonical_path`; when empty it is resolved here, once.
    pub fn new(declared_root: &str, canonical: &str, declared: bool) -> Self {
        let declared_root = declared_root_of(declared_root);
        let canonical_root = if canonical.trim().is_empty() {
            canonical_root(Some(&declared_root))
        } else {
            canonical.trim().trim_end_matches('/').to_string()
        };
        Self {
            declared_root,
            canonical_root,
            declared,
        }
    }

    /// Is the declared root a link (two forms exist)?
    pub fn has_alias(&self) -> bool {
        !self.declared_root.is_empty()
            && !self.canonical_root.is_empty()
            && self.declared_root != self.canonical_root
    }

    /// The root every stored path of this project starts with.
    pub fn stored_root(&self) -> &str {
        if self.declared {
            &self.declared_root
        } else if !self.canonical_root.is_empty() {
            &self.canonical_root
        } else {
            &self.declared_root
        }
    }

    /// Bring `path` (absolute, lexically normalized) to the stored form.
    ///
    /// Either known form of the root is rewritten to the stored one by prefix,
    /// at a `/` boundary. A path under a *third* alias — neither root
    /// (`/var/…` vs `/private/var/…` on macOS, a second link elsewhere) — is
    /// resolved and mapped from its real form. A path outside the project, or
    /// one that cannot be resolved, is kept as given: the policy never invents
    /// a path. Without a project, the canonical form.
    pub fn stored(&self, path: &str) -> String {
        if self.declared_root.is_empty() {
            return canonicalize_or_keep(path);
        }
        let target = self.stored_root();
        if strip_root(target, path).is_some() {
            return path.to_string();
        }
        if self.has_alias() {
            let other = if self.declared {
                &self.canonical_root
            } else {
                &self.declared_root
            };
            if let Some(rest) = strip_root(other, path) {
                return format!("{target}{rest}");
            }
        }
        match std::fs::canonicalize(path) {
            Ok(real) => {
                let real = real.to_string_lossy();
                match strip_root(&self.canonical_root, &real) {
                    Some(rest) => format!("{target}{rest}"),
                    None => path.to_string(),
                }
            }
            Err(_) => path.to_string(),
        }
    }

    /// The path relative to the project root, whichever form `path` is in.
    pub fn relative(&self, path: &str) -> Option<String> {
        relative_to_root(&self.declared_root, path)
            .or_else(|| relative_to_root(&self.canonical_root, path))
    }

    /// The uid of a file of this project — see [`file_uid`]. A path outside
    /// both roots keeps its absolute form in the identity (never invented).
    pub fn uid(&self, project_id: Option<Uuid>, path: &str) -> String {
        match project_id {
            Some(pid) => file_uid(
                Some(pid),
                &self.relative(path).unwrap_or_else(|| path.to_string()),
            ),
            None => file_uid(None, path),
        }
    }
}

/// How long a cached [`PathPolicy`] is trusted. `migrate file-paths --apply`
/// runs in another process and cannot invalidate a server's cache: this bound
/// is how long the server may keep writing the old vocabulary after the
/// marker is set — a sync in that window creates alias nodes the next
/// `--apply` (idempotent) renames.
pub const PATH_POLICY_TTL: Duration = Duration::from_secs(30);

/// Per-project cache of [`PathPolicy`]: one graph round trip per project and
/// TTL instead of one per file (single-file syncs) or per tool call (the skill
/// hook). Invalidated when the project is updated.
#[derive(Default)]
pub struct PathPolicyCache {
    entries: tokio::sync::RwLock<HashMap<Uuid, (PathPolicy, Instant)>>,
}

impl PathPolicyCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The policy of `project_id`, from the cache when fresh.
    pub async fn get(
        &self,
        store: &dyn crate::neo4j::traits::ProjectStore,
        project_id: Uuid,
    ) -> Result<PathPolicy> {
        if let Some((policy, at)) = self.entries.read().await.get(&project_id) {
            if at.elapsed() < PATH_POLICY_TTL {
                return Ok(policy.clone());
            }
        }
        let policy = store.path_policy(project_id).await?;
        self.entries
            .write()
            .await
            .insert(project_id, (policy.clone(), Instant::now()));
        Ok(policy)
    }

    pub async fn invalidate(&self, project_id: Uuid) {
        self.entries.write().await.remove(&project_id);
    }
}

/// A project root resolved ONCE, to check many paths against it.
///
/// The name-based ignore list ([`crate::orchestrator::IGNORED_PATH_SEGMENTS`])
/// cannot answer "is this file really in the repository": it excludes
/// directories *by name*, and a project reached through a link is walked with
/// `follow_links(true)`, so a symlink pointing outside the repository would be
/// indexed as if it were part of it. The durable rule is not a list of names —
/// it is this predicate, asked of the LINKS the walk meets (a plain entry under
/// a directory already inside the root cannot leave it).
pub struct RealRoot {
    declared: PathBuf,
    real: Option<PathBuf>,
}

impl RealRoot {
    pub fn new(root: &Path) -> Self {
        Self {
            declared: root.to_path_buf(),
            real: root.canonicalize().ok(),
        }
    }

    /// Is `path` really inside the root, once links are resolved?
    ///
    /// Never answers `false` on a resolution failure: an unreadable path is
    /// kept, because dropping files silently is worse than indexing one too many.
    pub fn contains(&self, path: &Path) -> bool {
        match (&self.real, path.canonicalize()) {
            (Some(real_root), Ok(real)) => real.starts_with(real_root),
            _ => path.starts_with(&self.declared),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> Uuid {
        Uuid::parse_str("00333b5f-2d0a-4467-9c98-155e55d2b7e5").unwrap()
    }

    #[test]
    fn same_project_and_relative_path_give_the_same_uid() {
        assert_eq!(
            file_uid(Some(project()), "src/neo4j/mod.rs"),
            file_uid(Some(project()), "src/neo4j/mod.rs")
        );
    }

    #[test]
    fn two_projects_never_share_a_uid() {
        let other = Uuid::parse_str("b40221d7-bb2d-40c8-ab09-cbaa8fe61106").unwrap();
        assert_ne!(
            file_uid(Some(project()), "src/main.rs"),
            file_uid(Some(other), "src/main.rs")
        );
    }

    /// Une identité qu'on doit pouvoir relire, pas une empreinte opaque. Pour un
    /// fichier sans projet, le backfill écrit `'-:' + f.path` — le `/` initial
    /// reste, sinon « une grammaire, deux sites » serait faux.
    #[test]
    fn the_uid_is_readable_and_names_its_project() {
        assert_eq!(
            file_uid(Some(project()), "src/neo4j/mod.rs"),
            format!("{}:src/neo4j/mod.rs", project())
        );
        assert_eq!(file_uid(None, "/tmp/a.rs"), "-:/tmp/a.rs");
    }

    #[test]
    fn a_declared_root_is_normalized_once_in_rust() {
        let home = dirs::home_dir().unwrap().to_string_lossy().to_string();
        assert_eq!(declared_root_of("~/repo/"), format!("{home}/repo"));
        assert_eq!(declared_root_of(" /data/proj/ "), "/data/proj");
        assert_eq!(declared_root_of("/"), "/");
        assert_eq!(declared_root_of(""), "");
    }

    #[test]
    fn a_canonical_root_is_resolved_and_never_empty_when_declared() {
        let here = canonical_root(Some("src"));
        assert!(here.ends_with("src"), "got {here}");
        assert_eq!(canonical_root(None), "");
        assert_eq!(canonical_root(Some("   ")), "");
        assert_eq!(
            canonical_root(Some("/definitely/not/a/real/path-42")),
            "/definitely/not/a/real/path-42"
        );
        assert_eq!(
            try_canonical_root(Some("/definitely/not/a/real/path-42")),
            None
        );
    }

    /// Des racines peuvent être stockées avec `~` : la canonique doit l'étendre,
    /// sinon `canonical_path = "~/…"` littéral ne préfixe aucun chemin stocké.
    #[test]
    fn a_tilde_root_is_expanded_before_being_resolved() {
        let home = dirs::home_dir().expect("a home directory");
        let resolved = canonical_root(Some("~"));
        assert!(!resolved.starts_with('~'), "got {resolved}");
        assert_eq!(
            resolved,
            home.canonicalize().unwrap().to_string_lossy().to_string()
        );
        let fallback = canonical_root(Some("~/definitely/not/a/real/path-42/"));
        assert!(fallback.starts_with(&home.to_string_lossy().to_string()));
        assert!(!fallback.ends_with('/'));
    }

    /// Un échec de résolution à CE démarrage (volume absent) n'écrase jamais
    /// une valeur stockée : elle a été calculée quand le dossier était là.
    #[test]
    fn a_failed_resolution_never_overwrites_a_stored_canonical_path() {
        assert_eq!(
            canonical_path_update("/real/repo", None, "/link/repo"),
            None
        );
        assert_eq!(
            canonical_path_update("", None, "/link/repo/").as_deref(),
            Some("/link/repo"),
            "nothing stored yet: the normalized declared fallback is written once"
        );
        assert_eq!(
            canonical_path_update("/real/repo", Some("/real/repo"), "/link/repo"),
            None
        );
        assert_eq!(
            canonical_path_update("/old/real", Some("/real/repo"), "/link/repo").as_deref(),
            Some("/real/repo")
        );
    }

    /// `/data/proj-old/x.rs` n'est pas sous `/data/proj` : sans la frontière
    /// `/`, le strip donnait `-old/x.rs` — le fichier d'un autre projet.
    #[test]
    fn a_root_only_matches_at_a_slash_boundary() {
        assert_eq!(relative_to_root("/data/proj", "/data/proj-old/x.rs"), None);
        assert_eq!(
            relative_to_root("/data/proj", "/data/proj/x.rs").as_deref(),
            Some("x.rs")
        );
        assert_eq!(relative_to_root("/data/proj", "/data/proj"), None);
        let p = PathPolicy::new("/link/proj", "/data/proj", true);
        assert_eq!(p.stored("/data/proj-old/x.rs"), "/data/proj-old/x.rs");
        assert_eq!(p.stored("/data/proj/x.rs"), "/link/proj/x.rs");
    }

    #[test]
    fn a_path_outside_the_root_has_no_relative_form() {
        assert_eq!(
            relative_to_root("/Users/triviere/repo", "/Users/triviere/other/a.rs"),
            None
        );
    }

    #[test]
    fn the_perimeter_is_the_real_path_not_the_name() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("repo");
        std::fs::create_dir_all(base.join("inside")).unwrap();
        let inside = base.join("inside/a.rs");
        std::fs::write(&inside, "fn a() {}").unwrap();
        let outside = tmp.path().join("outside.rs");
        std::fs::write(&outside, "fn b() {}").unwrap();

        let root = RealRoot::new(&base);
        assert!(root.contains(&inside));
        assert!(
            !root.contains(&base.join("../outside.rs")),
            "a path that only looks like it belongs must not pass"
        );
        // A path that cannot be resolved is kept, never dropped by accident.
        assert!(root.contains(&base.join("nope.rs")));
    }

    fn policy(declared: bool) -> PathPolicy {
        PathPolicy::new(
            "/Users/triviere/projects/project-orchestrator/backend",
            "/Users/triviere/.openclaw/workspace/skills/project-orchestrator",
            declared,
        )
    }

    /// Le verrou : un projet NON marqué garde son vocabulaire (canonique), quel
    /// que soit le chemin reçu ; un projet marqué écrit la forme déclarée.
    #[test]
    fn an_unmarked_project_keeps_its_canonical_vocabulary() {
        let p = policy(false);
        let declared = "/Users/triviere/projects/project-orchestrator/backend/src/main.rs";
        let canonical =
            "/Users/triviere/.openclaw/workspace/skills/project-orchestrator/src/main.rs";
        assert_eq!(
            p.stored(declared),
            canonical,
            "a declared input is re-mounted"
        );
        assert_eq!(p.stored(canonical), canonical, "a canonical input is kept");
        assert_eq!(p.stored("/elsewhere/a.rs"), "/elsewhere/a.rs");
        assert_eq!(p.stored_root(), p.canonical_root);
    }

    #[test]
    fn a_marked_project_writes_the_declared_form() {
        let p = policy(true);
        let declared = "/Users/triviere/projects/project-orchestrator/backend/src/main.rs";
        let canonical =
            "/Users/triviere/.openclaw/workspace/skills/project-orchestrator/src/main.rs";
        assert_eq!(p.stored(canonical), declared);
        assert_eq!(p.stored(declared), declared);
        assert_eq!(p.stored_root(), p.declared_root);
    }

    /// L'identité ne voit pas le montage : les deux formes donnent le même uid,
    /// que le projet soit marqué ou non — c'est ce que les chemins d'écriture
    /// calculent (en Rust) et ce que le backfill calcule (même fonction).
    #[test]
    fn the_uid_is_the_same_through_both_forms_and_both_policies() {
        let declared = "/Users/triviere/projects/project-orchestrator/backend/src/main.rs";
        let canonical =
            "/Users/triviere/.openclaw/workspace/skills/project-orchestrator/src/main.rs";
        let expected = format!("{}:src/main.rs", project());
        for p in [policy(false), policy(true)] {
            assert_eq!(p.uid(Some(project()), declared), expected);
            assert_eq!(p.uid(Some(project()), canonical), expected);
            assert_eq!(p.relative("/elsewhere/a.rs"), None);
            assert_eq!(
                p.uid(Some(project()), "/elsewhere/a.rs"),
                format!("{}:/elsewhere/a.rs", project()),
                "outside both roots: the absolute path, never an invented relative one"
            );
        }
        assert_eq!(PathPolicy::none().uid(None, "/tmp/a.rs"), "-:/tmp/a.rs");
        // A tilde root is normalized once, in the policy.
        let home = dirs::home_dir().unwrap().to_string_lossy().to_string();
        let p = PathPolicy::new("~/repo/", "/real/repo", true);
        assert_eq!(p.declared_root, format!("{home}/repo"));
        assert_eq!(
            p.uid(Some(project()), &format!("{home}/repo/src/a.rs")),
            format!("{}:src/a.rs", project())
        );
    }

    /// Sans projet, la forme stockée est la canonique : l'invariant des syncs
    /// hors projet (un fichier, un nœud, quel que soit le montage) ne dépend
    /// plus de `normalize_path`.
    #[cfg(unix)]
    #[test]
    fn without_a_project_the_stored_form_is_canonical() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("a.rs"), "fn a() {}").unwrap();
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let none = PathPolicy::none();
        let via_link = none.stored(&format!("{}/a.rs", link.display()));
        let via_real = none.stored(&format!("{}/a.rs", real.display()));
        assert_eq!(via_link, via_real, "one file, one stored form");
        assert_eq!(
            via_real,
            real.canonicalize()
                .unwrap()
                .join("a.rs")
                .display()
                .to_string()
        );
        assert_eq!(
            none.stored("/nope/x.rs"),
            "/nope/x.rs",
            "unresolvable: kept"
        );
    }

    /// Un troisième alias (`/var` vs `/private/var` sur macOS, un second lien)
    /// est résolu depuis le chemin lui-même, dans les deux sens.
    #[cfg(unix)]
    #[test]
    fn a_third_alias_is_resolved_from_the_path_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        std::fs::create_dir_all(real.join("src")).unwrap();
        std::fs::write(real.join("src/a.rs"), "fn a() {}").unwrap();
        let link = tmp.path().join("link");
        let other = tmp.path().join("other");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        std::os::unix::fs::symlink(&real, &other).unwrap();
        let declared = link.to_string_lossy().to_string();
        let third = format!("{}/src/a.rs", other.display());

        let marked = PathPolicy::new(&declared, "", true);
        assert_eq!(marked.stored(&third), format!("{declared}/src/a.rs"));
        let unmarked = PathPolicy::new(&declared, "", false);
        assert_eq!(
            unmarked.stored(&third),
            format!("{}/src/a.rs", unmarked.canonical_root)
        );
    }

    #[tokio::test]
    async fn the_policy_cache_asks_the_store_once_per_project() {
        let store = crate::neo4j::mock::MockGraphStore::new();
        let mut project = crate::test_helpers::test_project_named("cached");
        project.root_path = "/tmp/cached-root".into();
        let pid = project.id;
        store.projects.write().await.insert(pid, project);
        let cache = PathPolicyCache::new();
        let reads = || store.store_reads.load(std::sync::atomic::Ordering::Relaxed);
        let before = reads();
        let a = cache.get(&store, pid).await.unwrap();
        let b = cache.get(&store, pid).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(reads() - before, 1, "the second call hits the cache");
        assert!(!a.declared);
        store.paths_declared.write().await.insert(pid, true);
        assert!(
            !cache.get(&store, pid).await.unwrap().declared,
            "still cached"
        );
        cache.invalidate(pid).await;
        assert!(cache.get(&store, pid).await.unwrap().declared);
    }
}
