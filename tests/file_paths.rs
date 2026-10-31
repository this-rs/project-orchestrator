//! The file-identity Cypher against a real Neo4j (T7, lot 3/3 — review 2):
//! adoption of a node at the same path, the uid backfill with a duplicate,
//! the rename with a `~` root, the twin merge, the marker discipline.
//!
//! These are the writes a reading of the Cypher cannot prove. Every scenario
//! is scoped to a fresh project id and its own directory; NEVER point it at a
//! live database. Run locally against a throwaway instance (APOC required, as
//! for `data_migrations`):
//!   docker run -d --rm --name po-t7 -p 17688:7687 -e NEO4J_AUTH=neo4j/testpassword \
//!     -e NEO4J_PLUGINS='["apoc"]' neo4j:5.26
//!   NEO4J_URI=bolt://localhost:17688 NEO4J_PASSWORD=testpassword cargo test --test file_paths
//!
//! The tests print a notice and return when no Neo4j is reachable;
//! `HARNESS_NEO4J_REQUIRED=1` turns that into a failure.
#[allow(unused_imports)]
use project_orchestrator::neo4j::traits::*;

use neo4rs::{query, Graph};
use project_orchestrator::neo4j::client::Neo4jClient;
use project_orchestrator::neo4j::file_uid::PathPolicy;
use project_orchestrator::neo4j::models::{FileNode, ProjectNode};
use std::sync::OnceLock;
use uuid::Uuid;

struct Env {
    client: Neo4jClient,
    raw: Graph,
}

async fn env() -> Option<Env> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17688".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let connected = async {
        let client = Neo4jClient::new(&uri, &user, &password).await.ok()?;
        let raw = Graph::new(&uri, &user, &password).await.ok()?;
        Some(Env { client, raw })
    }
    .await;
    if connected.is_none() {
        assert!(
            std::env::var("HARNESS_NEO4J_REQUIRED").is_err(),
            "HARNESS_NEO4J_REQUIRED is set but no Neo4j answers at {uri}"
        );
        eprintln!("Skipping file_paths tests: no Neo4j at {uri}");
    }
    connected
}

/// One HOME for the whole test binary, so `~/…` roots resolve into a sandbox.
/// Set once, before any thread reads it.
fn sandbox_home() -> &'static std::path::Path {
    static HOME: OnceLock<tempfile::TempDir> = OnceLock::new();
    let dir = HOME.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", dir.path());
        dir
    });
    dir.path()
}

/// A repository reached through a link: `(real, link)` as absolute strings,
/// the real one canonical. Both under the sandbox HOME.
fn linked_repo(name: &str) -> (String, String) {
    let home = sandbox_home();
    let real = home.join(format!("{name}-real"));
    std::fs::create_dir_all(real.join("src")).unwrap();
    std::fs::write(real.join("src/a.rs"), "fn a() {}").unwrap();
    let link = home.join(format!("{name}-link"));
    std::os::unix::fs::symlink(&real, &link).unwrap();
    (
        real.canonicalize().unwrap().display().to_string(),
        link.display().to_string(),
    )
}

async fn count(raw: &Graph, cypher: &str, p: &str) -> i64 {
    let mut r = raw.execute(query(cypher).param("p", p)).await.unwrap();
    r.next().await.unwrap().unwrap().get::<i64>("c").unwrap()
}

async fn strings(raw: &Graph, cypher: &str, p: &str) -> Vec<String> {
    let mut r = raw.execute(query(cypher).param("p", p)).await.unwrap();
    let mut out = Vec::new();
    while let Some(row) = r.next().await.unwrap() {
        out.push(row.get::<String>("v").unwrap());
    }
    out.sort();
    out
}

async fn opt_string(raw: &Graph, cypher: &str, p: &str) -> Option<String> {
    let mut r = raw.execute(query(cypher).param("p", p)).await.unwrap();
    r.next()
        .await
        .unwrap()
        .and_then(|row| row.get::<Option<String>>("v").unwrap())
}

/// A LEGACY project node, as the graph holds it before this series: raw
/// `root_path`, no `declared_path`, no `canonical_path`, no marker.
async fn legacy_project(raw: &Graph, slug: &str, root_path: &str) -> Uuid {
    let id = Uuid::new_v4();
    raw.run(
        query(
            "CREATE (:Project {id: $id, name: $slug, slug: $slug, root_path: $root,
                     created_at: datetime(), watch_enabled: false, profile: 'software'})",
        )
        .param("id", id.to_string())
        .param("slug", slug)
        .param("root", root_path),
    )
    .await
    .unwrap();
    id
}

fn file(path: &str, project: Option<Uuid>) -> FileNode {
    FileNode {
        path: path.to_string(),
        language: "rust".into(),
        hash: "h".into(),
        last_parsed: chrono::Utc::now(),
        project_id: project,
    }
}

async fn cleanup(raw: &Graph, project: Uuid) {
    raw.run(
        query(
            "MATCH (n) WHERE n.project_id = $p OR (n:Project AND n.id = $p)
             OPTIONAL MATCH (n)-[:CONTAINS]->(s) DETACH DELETE s, n",
        )
        .param("p", project.to_string()),
    )
    .await
    .unwrap();
}

/// Review 2, point 2 (and 1): a node already at the path, under another uid or
/// none, is ADOPTED by the write — never a second node, never a
/// `ConstraintValidationFailed` that fails the whole sync.
#[tokio::test]
async fn a_node_at_the_same_path_is_adopted_by_the_write() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };
    let (real, _link) = linked_repo("adopt");
    let path = format!("{real}/src/a.rs");

    // A project created by the server: born in the declared vocabulary.
    let mut project = ProjectNode {
        id: Uuid::new_v4(),
        name: "adopt".into(),
        slug: format!("adopt-{}", Uuid::new_v4()),
        root_path: real.clone(),
        description: None,
        created_at: chrono::Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: false,
        profile: Default::default(),
    };
    client.create_project(&project).await.unwrap();
    let p = project.id.to_string();

    // A legacy File at that path: no uid at all (never backfilled).
    raw.run(
        query("CREATE (:File {path: $path, project_id: $p, language: 'rust', hash: 'old'})")
            .param("path", path.clone())
            .param("p", p.clone()),
    )
    .await
    .unwrap();

    client
        .upsert_file(&file(&path, Some(project.id)))
        .await
        .unwrap();
    assert_eq!(
        count(
            &raw,
            "MATCH (f:File {path: $p}) RETURN count(f) AS c",
            &path
        )
        .await,
        1,
        "adopted, not duplicated"
    );
    assert_eq!(
        opt_string(&raw, "MATCH (f:File {path: $p}) RETURN f.uid AS v", &path).await,
        Some(format!("{p}:src/a.rs")),
        "the uid is computed in Rust against the relative path"
    );

    // The project is deleted and recreated (new uuid) over the same directory:
    // the node at the path changes owner and identity, the path constraint holds.
    project.id = Uuid::new_v4();
    project.slug = format!("adopt-{}", Uuid::new_v4());
    client.create_project(&project).await.unwrap();
    let q = project.id.to_string();
    client
        .batch_upsert_files(&[file(&path, Some(project.id))])
        .await
        .unwrap();
    assert_eq!(
        count(
            &raw,
            "MATCH (f:File {path: $p}) RETURN count(f) AS c",
            &path
        )
        .await,
        1
    );
    assert_eq!(
        opt_string(&raw, "MATCH (f:File {path: $p}) RETURN f.uid AS v", &path).await,
        Some(format!("{q}:src/a.rs"))
    );

    cleanup(&raw, Uuid::parse_str(&p).unwrap()).await;
    cleanup(&raw, project.id).await;
}

/// Review 2, point 4: two `File` of one project for one file (real + link,
/// indexed before this series) compute the same uid. The backfill writes it
/// once, marks the other `uid_conflict`, converges, and is idempotent — no
/// `file_uid` violation, no retry loop at every boot.
#[tokio::test]
async fn the_uid_backfill_records_a_duplicate_instead_of_failing() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };
    let (real, link) = linked_repo("dup");
    let pid = legacy_project(&raw, &format!("dup-{}", Uuid::new_v4()), &link).await;
    let p = pid.to_string();
    client.ensure_project_canonical_paths().await.unwrap();
    raw.run(
        query(
            "MATCH (p:Project {id: $p})
             CREATE (:File {path: $real + '/src/a.rs', project_id: $p, language: 'rust', hash: 'r'})
             CREATE (:File {path: $link + '/src/a.rs', project_id: $p, language: 'rust', hash: 'l'})",
        )
        .param("p", p.clone())
        .param("real", real.clone())
        .param("link", link.clone()),
    )
    .await
    .unwrap();

    let out = client
        .run_data_migration_scoped("2026-10-backfill-file-uid", pid)
        .await
        .unwrap();
    assert!(out.completed, "{out:?}");
    assert_eq!(out.processed, 2, "both rows were selected and written once");
    assert_eq!(out.deferred, 1, "one of them is a conflict");
    assert_eq!(
        strings(
            &raw,
            "MATCH (f:File {project_id: $p}) WHERE f.uid IS NOT NULL RETURN f.uid AS v",
            &p
        )
        .await,
        vec![format!("{p}:src/a.rs")],
        "the uid is held once"
    );
    assert_eq!(
        strings(
            &raw,
            "MATCH (f:File {project_id: $p}) WHERE f.uid_conflict IS NOT NULL RETURN f.uid_conflict AS v",
            &p
        )
        .await,
        vec![format!("{p}:src/a.rs")],
        "the duplicate is recorded, not written"
    );
    let again = client
        .run_data_migration_scoped("2026-10-backfill-file-uid", pid)
        .await
        .unwrap();
    assert!(again.completed);
    assert_eq!(
        again.processed, 0,
        "idempotent: the conflict left the selection"
    );

    cleanup(&raw, pid).await;
}

/// Review 2, points 1-2-3: a legacy project declared with `~`. The boot repair
/// writes the NORMALIZED `declared_path`; the rename rewrites `File.path`, the
/// symbols' `file_path`/`id`/`stable_key` and the anchors from it — no `~`
/// anywhere, the uid untouched, the note edge intact — and sets the marker
/// only once every pass succeeded.
#[tokio::test]
async fn the_rename_uses_the_normalized_declared_root_and_marks_last() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };
    let home = sandbox_home().display().to_string();
    let (real, link) = linked_repo("tilde");
    let tilde_root = format!("~/{}/", link.rsplit('/').next().unwrap());
    let slug = format!("tilde-{}", Uuid::new_v4());
    let pid = legacy_project(&raw, &slug, &tilde_root).await;
    let p = pid.to_string();

    // The alias graph: a File (uid set, as after the backfill), a Function with
    // the three path-bearing properties, a Note anchored on both, a file Anchor.
    raw.run(
        query(
            "CREATE (f:File {path: $real + '/src/a.rs', uid: $p + ':src/a.rs', project_id: $p,
                             language: 'rust', hash: 'h'})
             CREATE (fn:Function {id: $real + '/src/a.rs:a:1', file_path: $real + '/src/a.rs',
                                  stable_key: $real + '/src/a.rs::a', name: 'a', project_id: $p,
                                  line_start: 1, line_end: 1})
             CREATE (f)-[:CONTAINS]->(fn)
             CREATE (n:Note {id: randomUUID(), project_id: $p, title: 'note'})
             CREATE (n)-[:LINKED_TO]->(f)
             CREATE (n)-[:LINKED_TO]->(fn)
             CREATE (:Anchor {id: randomUUID(), session_id: 's-' + $p, target_type: 'file',
                              target_id: $real + '/src/a.rs', snapshot_path: $real + '/src/a.rs'})",
        )
        .param("real", real.clone())
        .param("p", p.clone()),
    )
    .await
    .unwrap();

    client.ensure_project_canonical_paths().await.unwrap();
    let plan = client.file_path_plan(&slug).await.unwrap();
    assert!(plan.has_alias(), "{plan:?}");
    assert_eq!(
        plan.declared_root, link,
        "tilde expanded, no trailing slash"
    );
    assert!(plan.declared_root.starts_with(&home));
    assert_eq!(plan.canonical_root, real);
    assert!(!plan.paths_declared);

    let outcomes = client.file_path_rename(&plan).await.unwrap();
    assert!(outcomes.iter().all(|o| o.error.is_none()), "{outcomes:?}");

    let paths = strings(
        &raw,
        "MATCH (n) WHERE n.project_id = $p AND (n:File OR n:Function)
         RETURN coalesce(n.path, n.file_path) AS v",
        &p,
    )
    .await;
    assert_eq!(
        paths,
        vec![format!("{link}/src/a.rs"), format!("{link}/src/a.rs")],
        "File.path and Function.file_path in the declared form"
    );
    assert!(paths.iter().all(|v| !v.contains('~')), "no literal tilde");
    assert_eq!(
        strings(
            &raw,
            "MATCH (n:Function {project_id: $p}) RETURN n.id AS v",
            &p
        )
        .await,
        vec![format!("{link}/src/a.rs:a:1")]
    );
    assert_eq!(
        strings(
            &raw,
            "MATCH (n:Function {project_id: $p}) RETURN n.stable_key AS v",
            &p
        )
        .await,
        vec![format!("{link}/src/a.rs::a")]
    );
    assert_eq!(
        strings(
            &raw,
            "MATCH (f:File {project_id: $p}) RETURN f.uid AS v",
            &p
        )
        .await,
        vec![format!("{p}:src/a.rs")],
        "the uid never moves"
    );
    assert_eq!(
        strings(
            &raw,
            "MATCH (a:Anchor {session_id: 's-' + $p}) RETURN a.target_id AS v",
            &p
        )
        .await,
        vec![format!("{link}/src/a.rs")]
    );
    assert_eq!(
        count(
            &raw,
            "MATCH (:Note {project_id: $p})-[r:LINKED_TO]->() RETURN count(r) AS c",
            &p
        )
        .await,
        2,
        "renamed in place: the knowledge edges survive"
    );
    assert_eq!(
        opt_string(
            &raw,
            "MATCH (pr:Project {id: $p}) RETURN toString(pr.paths_declared) AS v",
            &p
        )
        .await
        .as_deref(),
        Some("true"),
        "marked once every pass succeeded"
    );
    // From here the sync writes the declared form.
    let policy = client.path_policy(pid).await.unwrap();
    assert!(policy.declared);
    assert_eq!(
        policy.stored(&format!("{real}/src/a.rs")),
        format!("{link}/src/a.rs")
    );
    // Idempotent: a second run has nothing to do and keeps the marker.
    let again = client.file_path_rename(&plan).await.unwrap();
    assert!(again.iter().all(|o| o.processed == 0), "{again:?}");

    raw.run(query("MATCH (a:Anchor {session_id: 's-' + $p}) DELETE a").param("p", p.clone()))
        .await
        .unwrap();
    cleanup(&raw, pid).await;
}

/// Review 2, point 3 (merge, review 1 point 3): an alias node whose declared
/// twin already exists is FOLDED into the twin — its edges move, it disappears
/// — never skipped. And a project whose uid coverage is incomplete is refused
/// before any write, marker absent.
#[tokio::test]
async fn an_alias_with_a_twin_is_folded_and_an_incomplete_coverage_is_refused() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };
    let (real, link) = linked_repo("twin");
    let slug = format!("twin-{}", Uuid::new_v4());
    let pid = legacy_project(&raw, &slug, &link).await;
    let p = pid.to_string();
    client.ensure_project_canonical_paths().await.unwrap();

    // The alias File (uid NULL: the duplicate that lost the identity) and its
    // declared twin (uid held), each with its Function; a Note on the alias
    // Function only.
    raw.run(
        query(
            "CREATE (a:File {path: $real + '/src/a.rs', project_id: $p, language: 'rust', hash: 'old',
                             uid_conflict: $p + ':src/a.rs'})
             CREATE (t:File {path: $link + '/src/a.rs', uid: $p + ':src/a.rs', project_id: $p,
                             language: 'rust', hash: 'new'})
             CREATE (afn:Function {id: $real + '/src/a.rs:a:1', file_path: $real + '/src/a.rs',
                                   name: 'a', project_id: $p, line_start: 1, line_end: 1})
             CREATE (tfn:Function {id: $link + '/src/a.rs:a:1', file_path: $link + '/src/a.rs',
                                   name: 'a', project_id: $p, line_start: 1, line_end: 1})
             CREATE (a)-[:CONTAINS]->(afn)
             CREATE (t)-[:CONTAINS]->(tfn)
             CREATE (n:Note {id: randomUUID(), project_id: $p, title: 'on the alias'})
             CREATE (n)-[:LINKED_TO]->(afn)",
        )
        .param("real", real.clone())
        .param("link", link.clone())
        .param("p", p.clone()),
    )
    .await
    .unwrap();

    let plan = client.file_path_plan(&slug).await.unwrap();
    // 1/2 files carry a uid: refused, nothing written, marker absent.
    let refused = client.file_path_rename(&plan).await;
    assert!(refused.is_err(), "incomplete uid coverage must be refused");
    assert!(
        format!("{:#}", refused.unwrap_err()).contains("couverture uid incomplète"),
        "the refusal names its reason"
    );
    assert_eq!(
        count(
            &raw,
            "MATCH (f:File {project_id: $p}) RETURN count(f) AS c",
            &p
        )
        .await,
        2,
        "nothing was written"
    );
    assert_eq!(
        opt_string(
            &raw,
            "MATCH (pr:Project {id: $p}) RETURN toString(pr.paths_declared) AS v",
            &p
        )
        .await,
        None,
        "marker absent after a refusal"
    );

    // The operator resolves the coverage (here: the duplicate is what the merge
    // pass folds, so it is enough to let it carry a uid of its own).
    raw.run(
        query("MATCH (f:File {project_id: $p}) WHERE f.uid IS NULL SET f.uid = $p + ':alias:src/a.rs'")
            .param("p", p.clone()),
    )
    .await
    .unwrap();
    let outcomes = client.file_path_rename(&plan).await.unwrap();
    assert!(outcomes.iter().all(|o| o.error.is_none()), "{outcomes:?}");

    assert_eq!(
        strings(
            &raw,
            "MATCH (f:File {project_id: $p}) RETURN f.path AS v",
            &p
        )
        .await,
        vec![format!("{link}/src/a.rs")],
        "one File: the alias was folded into the twin"
    );
    assert_eq!(
        strings(
            &raw,
            "MATCH (f:File {project_id: $p}) RETURN f.uid AS v",
            &p
        )
        .await,
        vec![format!("{p}:src/a.rs")],
        "the twin's properties win"
    );
    assert_eq!(
        strings(
            &raw,
            "MATCH (n:Function {project_id: $p}) RETURN n.id AS v",
            &p
        )
        .await,
        vec![format!("{link}/src/a.rs:a:1")],
        "one Function"
    );
    assert_eq!(
        count(
            &raw,
            "MATCH (:Note {project_id: $p})-[:LINKED_TO]->(fn:Function {project_id: $p})
             WHERE fn.id STARTS WITH $p OR true RETURN count(fn) AS c",
            &p
        )
        .await,
        1,
        "the note's edge moved onto the surviving Function"
    );
    cleanup(&raw, pid).await;
}

/// Review 2, point 9: the migration needs APOC and says so before writing.
#[tokio::test]
async fn apoc_is_required_and_present_on_the_throwaway_instance() {
    let Some(Env { client, .. }) = env().await else {
        return;
    };
    client.require_apoc().await.unwrap();
}

/// Review 2, point 6 (and review 1): a sync WITHOUT a project stores the
/// canonical form — one file, one node, whatever the mount.
#[tokio::test]
async fn a_project_less_write_stores_the_canonical_form() {
    let Some(Env { client, raw }) = env().await else {
        return;
    };
    let (real, link) = linked_repo("orphan");
    let via_link = PathPolicy::none().stored(&format!("{link}/src/a.rs"));
    assert_eq!(via_link, format!("{real}/src/a.rs"));
    client.upsert_file(&file(&via_link, None)).await.unwrap();
    client.upsert_file(&file(&via_link, None)).await.unwrap();
    assert_eq!(
        count(
            &raw,
            "MATCH (f:File {path: $p}) RETURN count(f) AS c",
            &via_link
        )
        .await,
        1
    );
    assert_eq!(
        opt_string(
            &raw,
            "MATCH (f:File {path: $p}) RETURN f.uid AS v",
            &via_link
        )
        .await,
        Some(format!("-:{real}/src/a.rs")),
        "`-:` + the absolute path, leading slash kept"
    );
    raw.run(query("MATCH (f:File {path: $p}) DETACH DELETE f").param("p", via_link))
        .await
        .unwrap();
}
