//! Reading dependency manifests.
//!
//! The regular sync pipeline cannot supply this: `scan_files` keeps only files
//! whose extension maps to a `SupportedLanguage`, so `Cargo.toml`, `package.json`
//! and friends never reach the parser. Architecture derivation therefore reads
//! them in a pass of its own.
//!
//! Manifests are the only place where infrastructure appears. Neo4j, Meilisearch
//! and NATS are not files in the workspace — they exist solely as declared
//! dependencies, which is why a derivation built on the code graph alone would
//! silently lose every datastore in the system.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Package ecosystem a manifest belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ecosystem {
    Cargo,
    Npm,
}

impl Ecosystem {
    pub fn as_str(self) -> &'static str {
        match self {
            Ecosystem::Cargo => "cargo",
            Ecosystem::Npm => "npm",
        }
    }
}

/// Where a dependency is resolved from. Git and path sources are what let us
/// recognise that one workspace project depends on another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencySource {
    Registry,
    Git { url: String },
    Path { path: String },
}

/// One dependency as declared in a manifest, with enough provenance to show the
/// user where a derived edge came from. A graph nobody typed is only believable
/// if every edge can be traced back to a line of a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestDependency {
    pub name: String,
    pub ecosystem: Ecosystem,
    /// Manifest path, relative to the project root.
    pub manifest_path: String,
    /// 1-based line of the declaration, for provenance display.
    pub line: u32,
    /// Cargo `optional = true` — the dependency sits behind a feature flag.
    pub optional: bool,
    pub source: DependencySource,
}

/// Cargo dependency value: either `foo = "1.0"` or `foo = { ... }`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CargoDep {
    /// The version string is never read — the variant exists so serde can tell the
    /// shorthand `foo = "1.0"` apart from the table form. Versions come from
    /// compose image tags, which pin what actually runs.
    Version(#[allow(dead_code)] String),
    Detailed {
        #[serde(default)]
        git: Option<String>,
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        optional: bool,
    },
}

#[derive(Debug, Deserialize)]
struct CargoManifest {
    #[serde(default)]
    dependencies: BTreeMap<String, CargoDep>,
    #[serde(default)]
    workspace: Option<CargoWorkspace>,
    #[serde(default)]
    package: Option<CargoPackage>,
    #[serde(default)]
    lib: Option<toml::Value>,
    #[serde(default, rename = "bin")]
    bins: Vec<toml::Value>,
    // dev-dependencies and build-dependencies are deliberately absent: a test-only
    // dependency is not part of the deployed architecture. Treating them as such
    // would put every test harness on the diagram.
}

#[derive(Debug, Deserialize)]
struct CargoPackage {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CargoWorkspace {
    #[serde(default)]
    dependencies: BTreeMap<String, CargoDep>,
}

#[derive(Debug, Deserialize)]
struct NpmManifest {
    #[serde(default)]
    dependencies: BTreeMap<String, String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    // devDependencies excluded, same reasoning as Cargo dev-dependencies.
}

/// Find the 1-based line declaring `name`, so a derived edge can point at its source.
///
/// Done by scanning rather than through a spanned parse: it handles `foo = ...`,
/// `"foo" = ...` and `[dependencies.foo]` uniformly, and a wrong line number only
/// degrades a tooltip — it never changes the topology.
fn find_declaration_line(raw: &str, name: &str) -> u32 {
    let quoted = format!("\"{}\"", name);
    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        let is_assignment = trimmed
            .strip_prefix(name)
            .or_else(|| trimmed.strip_prefix(&quoted))
            .map(|rest| rest.trim_start().starts_with('='))
            .unwrap_or(false);
        let is_section = trimmed.starts_with('[') && trimmed.ends_with(&format!(".{}]", name));
        if is_assignment || is_section {
            return idx as u32 + 1;
        }
    }
    0
}

fn cargo_dep_to_entry(
    name: &str,
    dep: &CargoDep,
    manifest_path: &str,
    raw: &str,
) -> ManifestDependency {
    let (source, optional) = match dep {
        CargoDep::Version(_) => (DependencySource::Registry, false),
        CargoDep::Detailed {
            git,
            path,
            optional,
        } => {
            let source = if let Some(url) = git {
                DependencySource::Git { url: url.clone() }
            } else if let Some(p) = path {
                DependencySource::Path { path: p.clone() }
            } else {
                DependencySource::Registry
            };
            (source, *optional)
        }
    };

    ManifestDependency {
        name: name.to_string(),
        ecosystem: Ecosystem::Cargo,
        manifest_path: manifest_path.to_string(),
        line: find_declaration_line(raw, name),
        optional,
        source,
    }
}

/// Parse a `Cargo.toml`. Both `[dependencies]` and `[workspace.dependencies]` are
/// read: a virtual workspace manifest declares everything in the latter and has no
/// `[dependencies]` at all (the nexus repo is exactly that shape).
pub fn parse_cargo_manifest(
    raw: &str,
    manifest_path: &str,
) -> anyhow::Result<Vec<ManifestDependency>> {
    let parsed: CargoManifest = toml::from_str(raw)?;
    let mut out = Vec::new();

    for (name, dep) in &parsed.dependencies {
        out.push(cargo_dep_to_entry(name, dep, manifest_path, raw));
    }
    if let Some(ws) = &parsed.workspace {
        for (name, dep) in &ws.dependencies {
            // A crate listed in both inherits the workspace entry; keep one.
            if out.iter().any(|d| d.name == *name) {
                continue;
            }
            out.push(cargo_dep_to_entry(name, dep, manifest_path, raw));
        }
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Parse a `package.json`.
pub fn parse_npm_manifest(
    raw: &str,
    manifest_path: &str,
) -> anyhow::Result<Vec<ManifestDependency>> {
    let parsed: NpmManifest = serde_json::from_str(raw)?;
    let mut out: Vec<ManifestDependency> = parsed
        .dependencies
        .keys()
        .map(|name| ManifestDependency {
            name: name.clone(),
            ecosystem: Ecosystem::Npm,
            manifest_path: manifest_path.to_string(),
            line: find_declaration_line(raw, name),
            optional: false,
            source: DependencySource::Registry,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Read every manifest under `root` that we understand.
///
/// Depth is capped and several trees skipped, for two distinct reasons.
///
/// Vendored trees (`node_modules`, `target`) would describe the architecture of
/// every transitive package rather than of this project.
///
/// `examples/`, `benches/`, `tests/` and `fixtures/` are skipped because they
/// are not the architecture either. Measured on the this-rs workspace: including
/// them turned one project into 216 dependencies and 28 internal edges across 14
/// sample apps — a diagram of one project's internals, presented as the
/// architecture of the workspace.
pub fn read_project_manifests(root: &Path) -> Vec<ManifestDependency> {
    const MAX_DEPTH: usize = 3;
    const SKIP_DIRS: [&str; 10] = [
        "node_modules",
        "target",
        "vendor",
        ".git",
        "dist",
        "build",
        "examples",
        "benches",
        "tests",
        "fixtures",
    ];

    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(MAX_DEPTH)
        .into_iter()
        .filter_entry(|e| {
            !e.file_name()
                .to_str()
                .map(|n| SKIP_DIRS.contains(&n))
                .unwrap_or(false)
        })
        .filter_map(Result::ok)
    {
        let file_name = entry.file_name().to_string_lossy().to_string();
        if file_name != "Cargo.toml" && file_name != "package.json" {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .to_string();

        // A malformed manifest is skipped rather than failing the whole derivation:
        // one unparseable file must not erase the rest of the architecture.
        let parsed = if file_name == "Cargo.toml" {
            parse_cargo_manifest(&raw, &rel)
        } else {
            parse_npm_manifest(&raw, &rel)
        };
        match parsed {
            Ok(deps) => out.extend(deps),
            Err(e) => tracing::debug!(manifest = %rel, error = %e, "skipping unparseable manifest"),
        }
    }
    out
}

/// What the root manifest says about the project itself.
///
/// Enough to type it. Without this a library is indistinguishable from a service
/// and both end up as `Other`, which is how `this` (a framework) and Nexus SDK
/// (a library) came to sit beside real services on the diagram.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectFacts {
    pub package_name: Option<String>,
    pub description: Option<String>,
    /// Declares a `[lib]` target and no binary — nothing to deploy.
    pub is_library: bool,
    /// Declares a binary target.
    pub has_binary: bool,
    /// Depends on a browser UI framework.
    pub is_frontend: bool,
    /// Depends on a command-line argument parser AND serves nothing over HTTP.
    /// Both a CLI and a server are just `[[bin]]` in the manifest, and nearly
    /// every server parses arguments too — so the web framework has to veto.
    pub is_cli: bool,
    pub ecosystem: Option<Ecosystem>,
}

/// Read project-level facts from a root `Cargo.toml` or `package.json`.
pub fn read_project_facts(root: &Path) -> ProjectFacts {
    const CLI_CRATES: [&str; 4] = ["clap", "structopt", "argh", "gumdrop"];

    if let Ok(raw) = std::fs::read_to_string(root.join("Cargo.toml")) {
        if let Ok(parsed) = toml::from_str::<CargoManifest>(&raw) {
            let has_binary = !parsed.bins.is_empty() || root.join("src/main.rs").exists();
            let dep_names: Vec<&String> = parsed.dependencies.keys().collect();
            return ProjectFacts {
                package_name: parsed.package.as_ref().and_then(|p| p.name.clone()),
                description: parsed.package.as_ref().and_then(|p| p.description.clone()),
                is_library: (parsed.lib.is_some() || root.join("src/lib.rs").exists())
                    && !has_binary,
                has_binary,
                is_frontend: false,
                is_cli: has_binary
                    && dep_names.iter().any(|n| CLI_CRATES.contains(&n.as_str()))
                    && !dep_names
                        .iter()
                        .any(|n| super::catalogue::is_web_framework(n)),
                ecosystem: Some(Ecosystem::Cargo),
            };
        }
    }

    if let Ok(raw) = std::fs::read_to_string(root.join("package.json")) {
        if let Ok(parsed) = serde_json::from_str::<NpmManifest>(&raw) {
            let is_frontend = parsed
                .dependencies
                .keys()
                .any(|n| super::catalogue::is_frontend_framework(n));
            return ProjectFacts {
                package_name: parsed.name,
                description: parsed.description,
                is_library: false,
                has_binary: false,
                is_frontend,
                is_cli: false,
                ecosystem: Some(Ecosystem::Npm),
            };
        }
    }

    ProjectFacts::default()
}

/// Is this dependency a hard architectural edge, or merely a capability?
///
/// A feature-gated dependency is not something the project uses — it is something
/// the project *can* use. The this-rs framework declares `sqlx`, `mongodb` and
/// `neo4rs` all `optional = true` behind its `postgres`, `mongodb_backend` and
/// `neo4j` features; drawing three hard edges from it would assert that it talks
/// to three databases at once, which is simply false.
///
/// Such dependencies still belong on the diagram — "this speaks seven databases"
/// is a real property of a framework — but as optional edges, rendered dashed and
/// filtered out by default.
pub fn is_hard_dependency(dep: &ManifestDependency) -> bool {
    !dep.optional
}

/// Where a `path` dependency actually points, relative to the project root.
///
/// Three cases that look alike in a manifest and mean completely different things:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathTarget {
    /// Stays under the project root — an internal crate. Part of how the project
    /// is organised, not a component of the system. Counting these turned
    /// `this-examples` into `billing`, `catalog`, `inventory` and `test-data`:
    /// module structure mistaken for architecture.
    Internal,
    /// Resolves to the project root itself. Every sub-crate of the orchestrator
    /// carries `project-orchestrator = { path = "../.." }`; drawing it would loop
    /// the project onto itself.
    SelfRoot,
    /// Climbs above the project root, so it can reach another component —
    /// this-rs/this reaching `../../lab/grafeo/crates/obrain-core`.
    Outside,
}

/// Classify a `path` dependency by walking it from the manifest's own directory.
///
/// The test is the *lowest* depth reached, not the final one: `../../lab/grafeo`
/// climbs above the root and comes back down, and a final-depth check reads that
/// as having stayed inside.
pub fn classify_path_dependency(manifest_path: &str, dep_path: &str) -> PathTarget {
    let mut depth = Path::new(manifest_path)
        .parent()
        .map(|p| p.components().count())
        .unwrap_or(0) as i32;

    let mut lowest = depth;
    for component in Path::new(dep_path).components() {
        match component {
            std::path::Component::ParentDir => depth -= 1,
            std::path::Component::CurDir => {}
            _ => depth += 1,
        }
        lowest = lowest.min(depth);
    }

    if lowest < 0 {
        PathTarget::Outside
    } else if depth == 0 {
        PathTarget::SelfRoot
    } else {
        PathTarget::Internal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CARGO: &str = r#"
[package]
name = "demo"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
neo4rs = "0.8"
async-nats = "0.49"
nexus-claude = { git = "https://github.com/this-rs/nexus.git", rev = "c68f046" }
local-lib = { path = "../local-lib" }
fancy = { version = "1", optional = true }

[dev-dependencies]
axum-test = "18"
"#;

    #[test]
    fn parses_cargo_dependencies_with_sources() {
        let deps = parse_cargo_manifest(CARGO, "Cargo.toml").unwrap();
        let by_name = |n: &str| deps.iter().find(|d| d.name == n).cloned().unwrap();

        assert_eq!(by_name("neo4rs").source, DependencySource::Registry);
        assert_eq!(
            by_name("nexus-claude").source,
            DependencySource::Git {
                url: "https://github.com/this-rs/nexus.git".into()
            }
        );
        assert_eq!(
            by_name("local-lib").source,
            DependencySource::Path {
                path: "../local-lib".into()
            }
        );
        assert!(by_name("fancy").optional);
        assert!(!by_name("neo4rs").optional);
    }

    #[test]
    fn excludes_dev_dependencies() {
        // A dev-dependency is not deployed. Including it would put test harnesses
        // on the architecture diagram.
        let deps = parse_cargo_manifest(CARGO, "Cargo.toml").unwrap();
        assert!(
            !deps.iter().any(|d| d.name == "axum-test"),
            "dev-dependencies must not reach the topology"
        );
    }

    #[test]
    fn records_declaration_line_for_provenance() {
        let deps = parse_cargo_manifest(CARGO, "Cargo.toml").unwrap();
        let neo4rs = deps.iter().find(|d| d.name == "neo4rs").unwrap();
        // Line 7 of the literal above (leading newline makes line 1 empty).
        assert_eq!(neo4rs.line, 7);
        assert_eq!(neo4rs.manifest_path, "Cargo.toml");
    }

    #[test]
    fn reads_virtual_workspace_manifest() {
        // The nexus repo has no [dependencies] at all — everything sits under
        // [workspace.dependencies]. Reading only [dependencies] would see nothing.
        let raw = r#"
[workspace]
members = ["a", "b"]

[workspace.dependencies]
neo4rs = "0.8"
meilisearch-sdk = { version = "0.33" }
"#;
        let deps = parse_cargo_manifest(raw, "Cargo.toml").unwrap();
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.name == "neo4rs"));
        assert!(deps.iter().any(|d| d.name == "meilisearch-sdk"));
    }

    #[test]
    fn parses_npm_dependencies_and_excludes_dev() {
        let raw = r#"{
  "name": "frontend",
  "dependencies": {
    "react": "^19.0.0",
    "neo4j-driver": "^5.0.0"
  },
  "devDependencies": {
    "vitest": "^4.0.0"
  }
}"#;
        let deps = parse_npm_manifest(raw, "package.json").unwrap();
        assert_eq!(deps.len(), 2);
        assert!(deps.iter().any(|d| d.name == "react"));
        assert!(!deps.iter().any(|d| d.name == "vitest"));
        assert_eq!(deps[0].ecosystem, Ecosystem::Npm);
    }

    #[test]
    fn finds_line_for_dotted_section_form() {
        let raw = "[dependencies]\nfoo = \"1\"\n\n[dependencies.bar]\nversion = \"2\"\n";
        assert_eq!(find_declaration_line(raw, "foo"), 2);
        assert_eq!(find_declaration_line(raw, "bar"), 4);
    }

    #[test]
    fn feature_gated_dependency_is_not_a_hard_edge() {
        // Taken from this-rs/this/Cargo.toml: three databases, all optional,
        // each behind a feature. The framework supports them; it does not use
        // them. Three hard edges here would assert something untrue.
        let raw = r#"
[dependencies]
sqlx = { version = "0.8", optional = true, default-features = false }
mongodb = { version = "3", optional = true }
neo4rs = { version = "0.8", optional = true }
axum = "0.8"
"#;
        let deps = parse_cargo_manifest(raw, "Cargo.toml").unwrap();
        let hard: Vec<_> = deps
            .iter()
            .filter(|d| is_hard_dependency(d))
            .map(|d| d.name.as_str())
            .collect();
        assert_eq!(hard, vec!["axum"]);

        // They are still recorded — "speaks seven databases" is a real property.
        assert_eq!(deps.len(), 4);
    }

    #[test]
    fn internal_path_dependency_is_not_a_component() {
        // this-examples: examples/rest depends on ../../crates/billing. That
        // climbs two levels and descends two — it never leaves the project.
        assert_eq!(
            classify_path_dependency("examples/rest/Cargo.toml", "../../crates/billing"),
            PathTarget::Internal
        );
        assert_eq!(
            classify_path_dependency("crates/test-data/Cargo.toml", "../billing"),
            PathTarget::Internal
        );
    }

    #[test]
    fn path_climbing_above_the_root_can_reach_another_component() {
        // this-rs/this depends on ../../lab/grafeo/crates/obrain-core. It climbs
        // above the root and descends again: judging by the FINAL depth would
        // wrongly call this internal — the lowest point reached is what counts.
        assert_eq!(
            classify_path_dependency("Cargo.toml", "../../lab/grafeo/crates/obrain-core"),
            PathTarget::Outside
        );
    }

    #[test]
    fn sub_crate_pointing_at_the_root_is_a_self_reference() {
        // Every orchestrator sub-crate carries project-orchestrator = { path = "../.." }.
        // Treated as an edge, the project depends on itself.
        assert_eq!(
            classify_path_dependency("crates/neural-routing-nn/Cargo.toml", "../.."),
            PathTarget::SelfRoot
        );
    }

    #[test]
    fn sample_and_bench_trees_are_not_scanned() {
        // Guarding the skip list itself: adding examples/ back would reintroduce
        // the 14-app hairball measured on this-examples.
        let dir = std::env::temp_dir().join(format!("arch-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("examples/demo")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[dependencies]\nneo4rs = \"0.8\"\n").unwrap();
        std::fs::write(
            dir.join("examples/demo/Cargo.toml"),
            "[dependencies]\nmongodb = \"3\"\n",
        )
        .unwrap();

        let deps = read_project_manifests(&dir);
        assert!(deps.iter().any(|d| d.name == "neo4rs"));
        assert!(
            !deps.iter().any(|d| d.name == "mongodb"),
            "examples/ must not contribute to the architecture"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_dependency_line_is_zero_not_a_wrong_line() {
        // Better to show no line than to point the user at an unrelated one.
        assert_eq!(
            find_declaration_line("[dependencies]\nfoo = \"1\"\n", "bar"),
            0
        );
    }
}
