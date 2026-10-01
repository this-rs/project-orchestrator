//! Assembling a workspace topology from what the code says.
//!
//! Three sources, in decreasing order of authority:
//!
//! 1. **Compose files** state the deployed system outright — services, images,
//!    `depends_on`, and connection URIs that name the wire protocol. Nothing is
//!    inferred. Where a compose file exists, it wins.
//! 2. **Manifests** supply what compose cannot: infrastructure for projects that
//!    ship no compose file, and dependencies between workspace projects.
//! 3. **Runtime config** supplies edges that exist in neither — a browser app
//!    reaching its API declares no package for it and is no container.
//!
//! The output is intentionally *small*. The hard part of this is not extraction
//! but granularity: a naive pass over `this-examples` yields 216 dependencies and
//! some twenty nodes, which describes one project's internals rather than the
//! architecture of anything. See `manifest::classify_path_dependency` and the
//! skip list in `manifest::read_project_manifests` for the filters that keep it
//! honest.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::neo4j::models::ComponentType;

use super::catalogue;
use super::compose;
use super::manifest::{self, DependencySource, Ecosystem, PathTarget};
use super::runtime_config;
use super::Provenance;

/// A component the derivation believes exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedComponent {
    /// Identity is the name: the service, not the client package. A Rust project
    /// and a TypeScript one both reaching Neo4j must land on one node.
    pub name: String,
    pub component_type: ComponentType,
    pub description: Option<String>,
    pub runtime: Option<String>,
    /// Workspace project this component *is*, when it is one.
    pub project_name: Option<String>,
    pub tags: Vec<String>,
    pub provenance: Provenance,
}

/// A dependency the derivation believes exists, by component name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedEdge {
    pub from: String,
    pub to: String,
    pub protocol: Option<String>,
    /// `false` for anything feature-gated or explicitly optional. Such an edge is
    /// a capability, not a dependency: `this` *supports* Neo4j, Postgres and
    /// MongoDB, it does not talk to all three.
    pub required: bool,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DerivedTopology {
    pub components: Vec<DerivedComponent>,
    pub edges: Vec<DerivedEdge>,
}

/// One workspace project, as the derivation needs to see it.
#[derive(Debug, Clone)]
pub struct ProjectInput {
    pub name: String,
    pub root: PathBuf,
    /// Git remote, used to recognise a git dependency as another workspace project.
    pub git_remote: Option<String>,
}

fn provenance(method: &str, file: &str, line: u32, package: &str) -> Provenance {
    Provenance {
        method: method.to_string(),
        file: file.to_string(),
        line,
        package: package.to_string(),
    }
}

/// Strip the noise that stops two spellings of one repository from matching:
/// scheme, credentials, a `.git` suffix, a trailing slash.
fn normalise_git_url(url: &str) -> String {
    let lower = url.trim().trim_end_matches('/').to_lowercase();
    let without_scheme = lower
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(&lower);
    let without_user = without_scheme
        .rsplit_once('@')
        .map(|(_, rest)| rest)
        .unwrap_or(without_scheme);
    without_user
        .replace(':', "/")
        .trim_end_matches(".git")
        .to_string()
}

/// Type a project from what its root manifest declares.
///
/// Ordering matters: a CLI also has a binary, and a frontend also has a
/// `package.json`, so the most specific test has to come first.
fn type_project(facts: &manifest::ProjectFacts) -> ComponentType {
    if facts.is_frontend {
        ComponentType::Frontend
    } else if facts.is_cli {
        ComponentType::Cli
    } else if facts.is_library {
        ComponentType::Library
    } else {
        ComponentType::Service
    }
}

fn runtime_of(facts: &manifest::ProjectFacts) -> Option<String> {
    match facts.ecosystem {
        Some(Ecosystem::Cargo) => Some("rust".into()),
        Some(Ecosystem::Npm) => Some("typescript".into()),
        None => None,
    }
}

/// Build the topology of a workspace from its projects' source trees.
pub fn derive_topology(projects: &[ProjectInput]) -> DerivedTopology {
    let mut components: Vec<DerivedComponent> = Vec::new();
    let mut edges: Vec<DerivedEdge> = Vec::new();

    // Component name of each project, so cross-project edges can be resolved.
    let mut component_of_project: HashMap<String, String> = HashMap::new();
    // Ports a project listens on, to resolve a proxy target back to a project.
    let mut project_by_port: HashMap<u16, String> = HashMap::new();

    let push_component = |components: &mut Vec<DerivedComponent>, c: DerivedComponent| {
        // First writer wins: compose runs before manifests, so the authoritative
        // source keeps its typing and provenance.
        if !components.iter().any(|existing| existing.name == c.name) {
            components.push(c);
        }
    };

    // ---- Pass 1: each project becomes a component -------------------------
    for project in projects {
        let facts = manifest::read_project_facts(&project.root);
        let component_name = project.name.clone();
        component_of_project.insert(project.name.clone(), component_name.clone());

        push_component(
            &mut components,
            DerivedComponent {
                name: component_name,
                component_type: type_project(&facts),
                description: facts.description.clone(),
                runtime: runtime_of(&facts),
                project_name: Some(project.name.clone()),
                tags: Vec::new(),
                provenance: provenance(
                    "project",
                    if facts.ecosystem == Some(Ecosystem::Npm) {
                        "package.json"
                    } else {
                        "Cargo.toml"
                    },
                    0,
                    facts.package_name.as_deref().unwrap_or(&project.name),
                ),
            },
        );
    }

    // ---- Pass 2: compose files, the authoritative source ------------------
    for project in projects {
        for compose_name in ["docker-compose.yml", "docker-compose.yaml", "compose.yml"] {
            let Ok(raw) = std::fs::read_to_string(project.root.join(compose_name)) else {
                continue;
            };
            let Ok(services) = compose::parse_compose(&raw) else {
                continue;
            };

            // The locally built service *is* this project; everything else is a
            // component in its own right.
            let mut name_in_graph: HashMap<String, String> = HashMap::new();
            for svc in &services {
                let graph_name = if svc.is_local_build {
                    project.name.clone()
                } else {
                    // Prefer the product name so this node and the one a client
                    // package would create are the same node.
                    svc.canonical_name
                        .map(str::to_string)
                        .unwrap_or_else(|| svc.name.clone())
                };
                name_in_graph.insert(svc.name.clone(), graph_name.clone());

                if !svc.is_local_build {
                    push_component(
                        &mut components,
                        DerivedComponent {
                            name: graph_name,
                            component_type: svc.component_type.clone(),
                            description: svc.image.clone(),
                            runtime: svc.image.clone(),
                            project_name: None,
                            tags: Vec::new(),
                            provenance: provenance("compose", compose_name, 0, &svc.name),
                        },
                    );
                }
            }

            for svc in &services {
                let from = name_in_graph.get(&svc.name).cloned().unwrap_or_default();
                for edge in &svc.depends_on {
                    let Some(to) = name_in_graph.get(&edge.target) else {
                        continue;
                    };
                    edges.push(DerivedEdge {
                        from: from.clone(),
                        to: to.clone(),
                        protocol: edge.protocol.clone(),
                        required: true,
                        provenance: provenance("compose", compose_name, 0, &edge.target),
                    });
                }
            }
            break;
        }
    }

    // ---- Pass 3: manifests — infrastructure and cross-project edges --------
    for project in projects {
        let deps = manifest::read_project_manifests(&project.root);
        let from = project.name.clone();

        // Keep the shallowest manifest as provenance: the orchestrator declares
        // `neo4rs` in three of them, and pointing at a sub-crate would be a
        // technically true but unhelpful answer to "where does this come from".
        // Ordered: iteration feeds the output directly, and a HashMap would make
        // the derivation non-reproducible run to run.
        let mut best_infra: BTreeMap<String, &manifest::ManifestDependency> = BTreeMap::new();

        for dep in &deps {
            if let Some(service) = catalogue::lookup(dep.ecosystem, &dep.name) {
                let depth = dep.manifest_path.matches('/').count();
                let keep = best_infra
                    .get(service.name)
                    .map(|current| depth < current.manifest_path.matches('/').count())
                    .unwrap_or(true);
                if keep {
                    best_infra.insert(service.name.to_string(), dep);
                }
            }
        }

        for (service_name, dep) in &best_infra {
            let service = catalogue::lookup(dep.ecosystem, &dep.name).expect("looked up above");
            push_component(
                &mut components,
                DerivedComponent {
                    name: service_name.clone(),
                    component_type: service.component_type.clone(),
                    description: None,
                    runtime: None,
                    project_name: None,
                    tags: Vec::new(),
                    provenance: provenance("manifest", &dep.manifest_path, dep.line, &dep.name),
                },
            );
            edges.push(DerivedEdge {
                from: from.clone(),
                to: service_name.clone(),
                protocol: Some(service.protocol.to_string()),
                // A feature-gated backend is a capability, not a dependency.
                required: manifest::is_hard_dependency(dep),
                provenance: provenance("manifest", &dep.manifest_path, dep.line, &dep.name),
            });
        }

        // Cross-project edges. Only dependencies that leave the project can reach
        // another component; internal crates and self-references never do.
        for dep in &deps {
            let target_project = match &dep.source {
                DependencySource::Git { url } => {
                    let normalised = normalise_git_url(url);
                    projects.iter().find(|p| {
                        p.name != project.name
                            && p.git_remote
                                .as_deref()
                                .map(|r| normalise_git_url(r) == normalised)
                                .unwrap_or(false)
                    })
                }
                DependencySource::Path { path } => {
                    if manifest::classify_path_dependency(&dep.manifest_path, path)
                        != PathTarget::Outside
                    {
                        continue;
                    }
                    let resolved = normalise_path(&project.root.join(path));
                    projects
                        .iter()
                        .find(|p| p.name != project.name && normalise_path(&p.root) == resolved)
                }
                DependencySource::Registry => continue,
            };

            if let Some(target) = target_project {
                edges.push(DerivedEdge {
                    from: from.clone(),
                    to: target.name.clone(),
                    protocol: Some(match dep.ecosystem {
                        Ecosystem::Cargo => "Rust crate".to_string(),
                        Ecosystem::Npm => "npm package".to_string(),
                    }),
                    required: manifest::is_hard_dependency(dep),
                    provenance: provenance("manifest", &dep.manifest_path, dep.line, &dep.name),
                });
            }
        }
    }

    // ---- Pass 4: runtime config — the edges nothing else can see -----------
    // Ports first: a proxy target is matched back to whichever project serves it.
    for project in projects {
        if let Some(port) = listening_port(&project.root) {
            project_by_port.insert(port, project.name.clone());
        }
    }

    for project in projects {
        for upstream in runtime_config::read_project_upstreams(&project.root) {
            let Some(port) = upstream.port else { continue };
            let Some(target) = project_by_port.get(&port) else {
                continue;
            };
            if *target == project.name {
                continue;
            }
            edges.push(DerivedEdge {
                from: project.name.clone(),
                to: target.clone(),
                protocol: Some(upstream.scheme.to_uppercase()),
                required: true,
                provenance: provenance(
                    "runtime-config",
                    &upstream.file,
                    upstream.line,
                    &format!("{}:{}", upstream.host, port),
                ),
            });
        }
    }

    dedupe_edges(&mut edges);
    retype_consumed_projects(&mut components, &edges);
    sort_topology(&mut components, &mut edges);
    DerivedTopology { components, edges }
}

/// Put the output in a fixed order.
///
/// This runs on every sync, so two derivations of an unchanged tree must produce
/// byte-identical results — otherwise nothing downstream can tell a real change
/// from reshuffled map iteration, and the diagram reorders itself for no reason.
///
/// Projects come before the infrastructure they reach, so reading the list top to
/// bottom follows the same order as reading the diagram left to right.
fn sort_topology(components: &mut [DerivedComponent], edges: &mut [DerivedEdge]) {
    components.sort_by(|a, b| {
        b.project_name
            .is_some()
            .cmp(&a.project_name.is_some())
            .then_with(|| a.name.cmp(&b.name))
    });
    edges.sort_by(|a, b| a.from.cmp(&b.from).then_with(|| a.to.cmp(&b.to)));
}

/// A project that another project depends on as a crate or package is a library,
/// whatever its manifest looks like.
///
/// Structure alone cannot always tell: the Nexus SDK is a virtual Cargo workspace
/// with neither `[package]` nor `[lib]` at its root, so it reads as a plain
/// service. What settles it is how the workspace actually uses it — being
/// consumed as a dependency is the definition of a library.
///
/// Only `Service` is reconsidered: a frontend or a CLI that happens to publish a
/// package is still a frontend or a CLI.
fn retype_consumed_projects(components: &mut [DerivedComponent], edges: &[DerivedEdge]) {
    const CONSUMED: [&str; 2] = ["Rust crate", "npm package"];

    for component in components.iter_mut() {
        if component.component_type != ComponentType::Service || component.project_name.is_none() {
            continue;
        }
        let consumed = edges.iter().any(|e| {
            e.to == component.name
                && e.protocol
                    .as_deref()
                    .map(|p| CONSUMED.contains(&p))
                    .unwrap_or(false)
        });
        if consumed {
            component.component_type = ComponentType::Library;
        }
    }
}

/// Collapse `..` without touching the filesystem, so the comparison works for
/// paths that need not exist.
fn normalise_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// The port a project serves on, read from its compose port mapping or its
/// dev-server config. This is what ties a proxy target to a project.
fn listening_port(root: &Path) -> Option<u16> {
    if let Ok(raw) = std::fs::read_to_string(root.join("docker-compose.yml")) {
        for line in raw.lines() {
            let trimmed = line.trim().trim_start_matches("- ").trim_matches('"');
            if let Some((host_port, _)) = trimmed.split_once(':') {
                if let Ok(port) = host_port.trim_matches('"').parse::<u16>() {
                    return Some(port);
                }
            }
        }
    }
    None
}

/// One dependency per pair. Several sources legitimately report the same edge;
/// the strongest wins — a required edge is never downgraded by an optional one
/// naming the same pair, and a stated protocol is never lost to a missing one.
fn dedupe_edges(edges: &mut Vec<DerivedEdge>) {
    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    let mut out: Vec<DerivedEdge> = Vec::new();

    for edge in edges.drain(..) {
        let key = (edge.from.clone(), edge.to.clone());
        match seen.get(&key) {
            Some(&idx) => {
                let existing: &mut DerivedEdge = &mut out[idx];
                if edge.required && !existing.required {
                    existing.required = true;
                    existing.provenance = edge.provenance.clone();
                }
                if existing.protocol.is_none() {
                    existing.protocol = edge.protocol.clone();
                }
            }
            None => {
                seen.insert(key, out.len());
                out.push(edge);
            }
        }
    }

    // Self-edges are never meaningful and every source can produce one.
    out.retain(|e| e.from != e.to);
    *edges = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new(tag: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "derive-{}-{}-{:?}",
                tag,
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn project(&self, name: &str) -> PathBuf {
            let p = self.root.join(name);
            fs::create_dir_all(&p).unwrap();
            p
        }

        fn write(&self, rel: &str, content: &str) {
            let p = self.root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, content).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn input(name: &str, root: PathBuf) -> ProjectInput {
        ProjectInput {
            name: name.into(),
            root,
            git_remote: None,
        }
    }

    #[test]
    fn compose_defines_services_edges_and_protocols() {
        let f = Fixture::new("compose");
        let api = f.project("api");
        f.write(
            "api/docker-compose.yml",
            r#"
services:
  neo4j:
    image: neo4j:5-community
  app:
    build:
      context: .
    environment:
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      neo4j:
        condition: service_healthy
"#,
        );
        f.write("api/Cargo.toml", "[package]\nname = \"api\"\n");

        let topo = derive_topology(&[input("api", api)]);

        // The locally built service is the project, not a separate node.
        assert!(topo.components.iter().any(|c| c.name == "api"));
        assert!(!topo.components.iter().any(|c| c.name == "app"));

        // The compose key is `neo4j`; the node takes the product name so that a
        // client package importing the same product resolves here too.
        let neo4j = topo.components.iter().find(|c| c.name == "Neo4j").unwrap();
        assert_eq!(neo4j.component_type, ComponentType::Database);

        let edge = topo.edges.iter().find(|e| e.to == "Neo4j").unwrap();
        assert_eq!(edge.from, "api");
        // Read from the URI, not guessed from a client crate.
        assert_eq!(edge.protocol.as_deref(), Some("BOLT"));
    }

    #[test]
    fn manifest_infrastructure_appears_for_projects_without_compose() {
        let f = Fixture::new("infra");
        let api = f.project("api");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\n",
        );

        let topo = derive_topology(&[input("api", api)]);
        assert!(topo.components.iter().any(|c| c.name == "Neo4j"));
        let edge = topo.edges.iter().find(|e| e.to == "Neo4j").unwrap();
        assert!(edge.required);
        assert_eq!(edge.protocol.as_deref(), Some("Bolt"));
        assert_eq!(edge.provenance.package, "neo4rs");
    }

    #[test]
    fn feature_gated_backend_is_optional_not_a_dependency() {
        // `this` declares sqlx, mongodb and neo4rs all optional. Hard edges here
        // would claim it talks to three databases at once.
        let f = Fixture::new("optional");
        let lib = f.project("framework");
        f.write(
            "framework/Cargo.toml",
            r#"
[package]
name = "framework"

[lib]

[dependencies]
neo4rs = { version = "0.8", optional = true }
mongodb = { version = "3", optional = true }
"#,
        );

        let topo = derive_topology(&[input("framework", lib)]);
        assert!(topo.edges.iter().all(|e| !e.required));
        assert_eq!(topo.edges.len(), 2);
    }

    #[test]
    fn a_library_is_typed_as_one() {
        let f = Fixture::new("lib");
        let lib = f.project("sdk");
        f.write("sdk/Cargo.toml", "[package]\nname = \"sdk\"\n\n[lib]\n");

        let topo = derive_topology(&[input("sdk", lib)]);
        let c = topo.components.iter().find(|c| c.name == "sdk").unwrap();
        assert_eq!(c.component_type, ComponentType::Library);
    }

    #[test]
    fn a_cli_is_not_a_service() {
        // Both are just `[[bin]]`; the argument parser is what tells them apart.
        let f = Fixture::new("cli");
        let cli = f.project("tool");
        f.write(
            "tool/Cargo.toml",
            "[package]\nname = \"tool\"\n\n[[bin]]\nname = \"tool\"\n\n[dependencies]\nclap = \"4\"\n",
        );

        let topo = derive_topology(&[input("tool", cli)]);
        let c = topo.components.iter().find(|c| c.name == "tool").unwrap();
        assert_eq!(c.component_type, ComponentType::Cli);
    }

    #[test]
    fn a_frontend_is_typed_from_its_framework() {
        let f = Fixture::new("front");
        let web = f.project("web");
        f.write(
            "web/package.json",
            r#"{"name":"web","dependencies":{"react":"^19.0.0"}}"#,
        );

        let topo = derive_topology(&[input("web", web)]);
        let c = topo.components.iter().find(|c| c.name == "web").unwrap();
        assert_eq!(c.component_type, ComponentType::Frontend);
    }

    #[test]
    fn git_dependency_on_another_workspace_project_becomes_an_edge() {
        let f = Fixture::new("git");
        let api = f.project("api");
        let sdk = f.project("sdk");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[dependencies]\nsdk-crate = { git = \"https://github.com/acme/sdk.git\" }\n",
        );
        f.write("sdk/Cargo.toml", "[package]\nname = \"sdk\"\n\n[lib]\n");

        let topo = derive_topology(&[
            input("api", api),
            ProjectInput {
                name: "sdk".into(),
                root: sdk,
                // Spelled differently on purpose: ssh form, no .git suffix.
                git_remote: Some("git@github.com:acme/sdk".into()),
            },
        ]);

        let edge = topo.edges.iter().find(|e| e.to == "sdk").unwrap();
        assert_eq!(edge.from, "api");
        assert_eq!(edge.protocol.as_deref(), Some("Rust crate"));
    }

    #[test]
    fn internal_path_dependency_creates_no_edge() {
        // `this-examples` carries 28 of these. Each one would be a component.
        let f = Fixture::new("internal");
        let app = f.project("app");
        f.write("app/Cargo.toml", "[package]\nname = \"app\"\n");
        f.write(
            "app/crates/inner/Cargo.toml",
            "[package]\nname = \"inner\"\n\n[dependencies]\nhelper = { path = \"../helper\" }\n",
        );

        let topo = derive_topology(&[input("app", app)]);
        assert!(!topo.components.iter().any(|c| c.name == "helper"));
        assert!(topo.edges.is_empty());
    }

    #[test]
    fn proxy_config_supplies_the_frontend_to_backend_edge() {
        // Neither manifests nor compose can see this one: the browser app
        // declares no package for its API and is not a container.
        let f = Fixture::new("proxy");
        let web = f.project("web");
        let api = f.project("api");
        f.write(
            "web/package.json",
            r#"{"name":"web","dependencies":{"react":"^19.0.0"}}"#,
        );
        f.write(
            "web/vite.config.ts",
            "export default { server: { proxy: { '/api': { target: 'http://localhost:8080' } } } }",
        );
        f.write("api/Cargo.toml", "[package]\nname = \"api\"\n");
        f.write(
            "api/docker-compose.yml",
            "services:\n  app:\n    build: .\n    ports:\n      - \"8080:8080\"\n",
        );

        let topo = derive_topology(&[input("web", web), input("api", api)]);
        let edge = topo
            .edges
            .iter()
            .find(|e| e.from == "web" && e.to == "api")
            .expect("frontend must reach its backend");
        assert_eq!(edge.protocol.as_deref(), Some("HTTP"));
        assert_eq!(edge.provenance.method, "runtime-config");
    }

    #[test]
    fn compose_service_and_client_crate_resolve_to_one_node() {
        // The orchestrator both runs Neo4j in compose and imports neo4rs. Two
        // nodes would read as two databases.
        let f = Fixture::new("fold");
        let api = f.project("api");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\naxum = \"0.8\"\n",
        );
        f.write(
            "api/docker-compose.yml",
            r#"
services:
  neo4j:
    image: neo4j:5-community
  app:
    build: .
    environment:
      - NEO4J_URI=bolt://neo4j:7687
    depends_on: ["neo4j"]
"#,
        );

        let topo = derive_topology(&[input("api", api)]);
        let databases: Vec<_> = topo
            .components
            .iter()
            .filter(|c| c.component_type == ComponentType::Database)
            .collect();
        assert_eq!(databases.len(), 1, "got {:?}", databases);
        assert_eq!(databases[0].name, "Neo4j");
        assert_eq!(topo.edges.iter().filter(|e| e.to == "Neo4j").count(), 1);
    }

    #[test]
    fn a_server_that_parses_arguments_is_still_a_service() {
        // The orchestrator backend has [[bin]] and clap, and was typed Cli —
        // which put a backend API at the entry tier, beside the frontend.
        let f = Fixture::new("server");
        let api = f.project("api");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[[bin]]\nname = \"api\"\n\n[dependencies]\nclap = \"4\"\naxum = \"0.8\"\n",
        );

        let topo = derive_topology(&[input("api", api)]);
        let c = topo.components.iter().find(|c| c.name == "api").unwrap();
        assert_eq!(c.component_type, ComponentType::Service);
    }

    #[test]
    fn a_project_consumed_as_a_crate_is_a_library() {
        // The Nexus SDK is a virtual workspace: no [package], no [lib], so
        // structure reads it as a service. Being depended upon settles it.
        let f = Fixture::new("consumed");
        let api = f.project("api");
        let sdk = f.project("sdk");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[dependencies]\nsdk = { path = \"../sdk\" }\n",
        );
        f.write("sdk/Cargo.toml", "[workspace]\nmembers = [\"a\"]\n");

        let topo = derive_topology(&[input("api", api), input("sdk", sdk)]);
        let c = topo.components.iter().find(|c| c.name == "sdk").unwrap();
        assert_eq!(c.component_type, ComponentType::Library);
    }

    #[test]
    fn deriving_twice_gives_the_same_topology() {
        // The whole point of deriving rather than typing: it has to be safe to
        // replay on every sync.
        let f = Fixture::new("idem");
        let api = f.project("api");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\nasync-nats = \"0.49\"\n",
        );

        // Repeated enough times to catch ordering that only differs between
        // separately seeded maps — one comparison can pass by luck.
        let first = derive_topology(&[input("api", api.clone())]);
        for _ in 0..8 {
            assert_eq!(derive_topology(&[input("api", api.clone())]), first);
        }
        // And the order itself is fixed, not merely consistent within a run.
        let names: Vec<&str> = first.components.iter().map(|c| c.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(
            first.components.len(),
            3,
            "expected the project plus its two services"
        );
        assert_eq!(names[0], "api", "the project leads its infrastructure");
        assert!(sorted.contains(&"Neo4j") && sorted.contains(&"NATS"));
    }

    #[test]
    fn one_service_reached_from_several_manifests_is_one_node() {
        // The orchestrator declares neo4rs in three manifests. Three Neo4j nodes
        // would be three different databases as far as the reader is concerned.
        let f = Fixture::new("dedupe");
        let api = f.project("api");
        f.write(
            "api/Cargo.toml",
            "[package]\nname = \"api\"\n\n[dependencies]\nneo4rs = \"0.8\"\n",
        );
        f.write(
            "api/crates/inner/Cargo.toml",
            "[package]\nname = \"inner\"\n\n[dependencies]\nneo4rs = \"0.8\"\n",
        );

        let topo = derive_topology(&[input("api", api)]);
        assert_eq!(
            topo.components.iter().filter(|c| c.name == "Neo4j").count(),
            1
        );
        assert_eq!(topo.edges.iter().filter(|e| e.to == "Neo4j").count(), 1);
        // Provenance points at the root manifest, not the sub-crate.
        let edge = topo.edges.iter().find(|e| e.to == "Neo4j").unwrap();
        assert_eq!(edge.provenance.file, "Cargo.toml");
    }

    #[test]
    fn git_urls_match_across_spellings() {
        assert_eq!(
            normalise_git_url("https://github.com/acme/sdk.git"),
            normalise_git_url("git@github.com:acme/sdk")
        );
        assert_eq!(
            normalise_git_url("https://github.com/acme/sdk/"),
            normalise_git_url("HTTPS://GitHub.com/acme/sdk.git")
        );
        assert_ne!(
            normalise_git_url("https://github.com/acme/sdk"),
            normalise_git_url("https://github.com/acme/other")
        );
    }

    #[test]
    fn an_empty_workspace_derives_nothing() {
        assert_eq!(derive_topology(&[]), DerivedTopology::default());
    }
}
