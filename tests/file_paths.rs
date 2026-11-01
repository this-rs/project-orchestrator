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

// ---------------------------------------------------------------------------
// `orchestrator migrate file-paths` — la commande `run`, de bout en bout.
// ---------------------------------------------------------------------------

use project_orchestrator::neo4j::path_migration;

/// A project whose declared root is a link to its canonical root, with the
/// canonical path stored. Returns `(slug, id, real, link)`.
async fn alias_project(env: &Env, name: &str) -> (String, Uuid, String, String) {
    let (real, link) = linked_repo(name);
    let slug = format!("{name}-{}", Uuid::new_v4());
    let pid = legacy_project(&env.raw, &slug, &link).await;
    env.client.ensure_project_canonical_paths().await.unwrap();
    (slug, pid, real, link)
}

/// A `File` (uid set, as after the backfill) in the project.
async fn put_file(raw: &Graph, pid: Uuid, path: &str, uid: &str) {
    raw.run(
        query(
            "CREATE (:File {path: $path, uid: $uid, project_id: $p, language: 'rust', hash: 'h'})",
        )
        .param("path", path)
        .param("uid", uid)
        .param("p", pid.to_string()),
    )
    .await
    .unwrap();
}

/// A `Note` of the project linked to the files at `paths`.
async fn put_note(raw: &Graph, pid: Uuid, title: &str, paths: &[&str]) {
    for path in paths {
        raw.run(
            query(
                "MERGE (n:Note {id: $id}) ON CREATE SET n.project_id = $p, n.title = $title
                 WITH n MATCH (f:File {path: $path}) CREATE (n)-[:LINKED_TO]->(f)",
            )
            .param("id", format!("{pid}-{title}"))
            .param("p", pid.to_string())
            .param("title", title)
            .param("path", *path),
        )
        .await
        .unwrap();
    }
}

async fn paths_of_files(raw: &Graph, pid: Uuid) -> Vec<String> {
    strings(
        raw,
        "MATCH (f:File {project_id: $p}) RETURN f.path AS v",
        &pid.to_string(),
    )
    .await
}

async fn marker(raw: &Graph, pid: Uuid) -> Option<String> {
    opt_string(
        raw,
        "MATCH (pr:Project {id: $p}) RETURN toString(pr.paths_declared) AS v",
        &pid.to_string(),
    )
    .await
}

/// Without `--apply` nothing is written, whatever `--purge-wl` says: the plan
/// and the measures are printed, the exit code is 0.
#[tokio::test]
async fn the_dry_run_writes_nothing_and_exits_zero() {
    let Some(env) = env().await else {
        return;
    };
    let (slug, pid, real, _link) = alias_project(&env, "dry").await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/.wl/br/src/a.rs"),
        &format!("{pid}:wl"),
    )
    .await;

    for purge in [false, true] {
        let code = path_migration::run(&env.client, &slug, false, purge)
            .await
            .unwrap();
        assert_eq!(code, 0, "purge_wl={purge}");
    }
    assert_eq!(
        paths_of_files(&env.raw, pid).await,
        vec![
            format!("{real}/.wl/br/src/a.rs"),
            format!("{real}/src/a.rs")
        ],
        "the dry-run renamed nothing and purged nothing"
    );
    assert_eq!(marker(&env.raw, pid).await, None, "no marker either");
    cleanup(&env.raw, pid).await;
}

/// A root that is not a link: nothing to rename, exit 0, before `--apply`
/// is even looked at.
#[tokio::test]
async fn a_root_that_is_not_a_link_has_nothing_to_rename() {
    let Some(env) = env().await else {
        return;
    };
    let (real, _link) = linked_repo("plain");
    let slug = format!("plain-{}", Uuid::new_v4());
    let pid = legacy_project(&env.raw, &slug, &real).await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    let code = path_migration::run(&env.client, &slug, true, true)
        .await
        .unwrap();
    assert_eq!(code, 0);
    assert_eq!(marker(&env.raw, pid).await, None, "apply did not run");
    assert_eq!(
        paths_of_files(&env.raw, pid).await,
        vec![format!("{real}/src/a.rs")]
    );
    cleanup(&env.raw, pid).await;
}

/// An unknown slug is an error, not a silent exit 0.
#[tokio::test]
async fn an_unknown_slug_is_an_error() {
    let Some(env) = env().await else {
        return;
    };
    let err = path_migration::run(
        &env.client,
        &format!("nope-{}", Uuid::new_v4()),
        false,
        false,
    )
    .await
    .unwrap_err();
    assert!(format!("{err:#}").contains("aucun projet"), "{err:#}");
}

/// `--apply` renames and marks; without `--purge-wl` the `.wl` copy stays.
/// A second run, now that the marker is set, still exits 0 (idempotent).
#[tokio::test]
async fn apply_renames_marks_and_keeps_the_wl_copies_without_purge() {
    let Some(env) = env().await else {
        return;
    };
    let (slug, pid, real, link) = alias_project(&env, "apply").await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/.wl/br/src/a.rs"),
        &format!("{pid}:wl"),
    )
    .await;

    assert_eq!(
        path_migration::run(&env.client, &slug, true, false)
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        paths_of_files(&env.raw, pid).await,
        vec![
            format!("{real}/.wl/br/src/a.rs"),
            format!("{link}/src/a.rs")
        ],
        "the file moved to the declared root, the .wl copy was left alone"
    );
    assert_eq!(marker(&env.raw, pid).await.as_deref(), Some("true"));
    // The plan now reads the marker.
    let plan = env.client.file_path_plan(&slug).await.unwrap();
    assert!(plan.paths_declared);
    assert_eq!(
        path_migration::run(&env.client, &slug, true, false)
            .await
            .unwrap(),
        0
    );
    cleanup(&env.raw, pid).await;
}

/// `--apply --purge-wl`: the knowledge of the `.wl` copy is re-pointed onto its
/// declared twin BEFORE the copy is deleted; the note keeps an anchor.
#[tokio::test]
async fn purge_rehomes_the_knowledge_of_a_wl_copy_then_deletes_it() {
    let Some(env) = env().await else {
        return;
    };
    let (slug, pid, real, link) = alias_project(&env, "purge").await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/.wl/br/src/a.rs"),
        &format!("{pid}:wl"),
    )
    .await;
    // A note whose ONLY anchor is the .wl copy; its twin is the renamed file.
    put_note(
        &env.raw,
        pid,
        "only-on-wl",
        &[&format!("{real}/.wl/br/src/a.rs")],
    )
    .await;

    let code = path_migration::run(&env.client, &slug, true, true)
        .await
        .unwrap();
    assert_eq!(code, 0, "the purge gate opens: the twin exists");
    assert_eq!(
        paths_of_files(&env.raw, pid).await,
        vec![format!("{link}/src/a.rs")],
        "the .wl copy is gone, the real file is under the declared root"
    );
    assert_eq!(
        strings(
            &env.raw,
            "MATCH (:Note {project_id: $p})-[:LINKED_TO]->(f:File) RETURN f.path AS v",
            &pid.to_string()
        )
        .await,
        vec![format!("{link}/src/a.rs")],
        "the note was re-pointed onto the twin, not orphaned"
    );
    cleanup(&env.raw, pid).await;
}

/// The gate refuses a purge that would orphan a note: a `.wl` copy without a
/// twin is the note's last anchor. Exit 2, the copy is still there.
#[tokio::test]
async fn the_purge_gate_refuses_when_a_note_would_lose_its_last_anchor() {
    let Some(env) = env().await else {
        return;
    };
    let (slug, pid, real, link) = alias_project(&env, "refuse").await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    // This copy has NO twin (`src/lonely.rs` exists nowhere else).
    put_file(
        &env.raw,
        pid,
        &format!("{real}/.wl/br/src/lonely.rs"),
        &format!("{pid}:lonely"),
    )
    .await;
    put_note(
        &env.raw,
        pid,
        "lonely",
        &[&format!("{real}/.wl/br/src/lonely.rs")],
    )
    .await;

    let code = path_migration::run(&env.client, &slug, true, true)
        .await
        .unwrap();
    assert_eq!(code, 2, "refused");
    assert_eq!(
        paths_of_files(&env.raw, pid).await,
        vec![
            format!("{real}/.wl/br/src/lonely.rs"),
            format!("{link}/src/a.rs")
        ],
        "renamed, but nothing was purged"
    );
    assert_eq!(
        count(
            &env.raw,
            "MATCH (:Note {project_id: $p})-[r:LINKED_TO]->() RETURN count(r) AS c",
            &pid.to_string()
        )
        .await,
        1,
        "the note keeps its edge"
    );
    cleanup(&env.raw, pid).await;
}

/// A pass that fails leaves NO marker, says to replay, and the replay (once the
/// cause is gone) completes and marks. The cause here: a unique constraint on
/// `Module.stable_key` that the rewritten key collides with.
#[tokio::test]
async fn a_failed_pass_removes_the_marker_and_the_replay_completes() {
    let Some(env) = env().await else {
        return;
    };
    let (slug, pid, real, link) = alias_project(&env, "failpass").await;
    let p = pid.to_string();
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    let constraint = format!("tmp_module_key_{}", p.replace('-', "_"));
    // The squatter belongs to nobody: the rename never touches it.
    raw_run(
        &env.raw,
        "CREATE (:Module {id: 'squat-' + $p, file_path: '/elsewhere/m.rs',
                          stable_key: $link + '/src/m.rs::m'})",
        &p,
        &link,
        &real,
    )
    .await;
    raw_run(
        &env.raw,
        "CREATE (:Module {id: $real + '/src/m.rs:m:1', file_path: $real + '/src/m.rs',
                          stable_key: $real + '/src/m.rs::m', project_id: $p})",
        &p,
        &link,
        &real,
    )
    .await;
    env.raw
        .run(query(&format!(
            "CREATE CONSTRAINT {constraint} IF NOT EXISTS FOR (m:Module) REQUIRE m.stable_key IS UNIQUE"
        )))
        .await
        .unwrap();

    let outcome = path_migration::run(&env.client, &slug, true, false).await;
    let _ = env
        .raw
        .run(query(&format!("DROP CONSTRAINT {constraint} IF EXISTS")))
        .await;
    let err = outcome.expect_err("the Module pass cannot rewrite the key");
    let text = format!("{err:#}");
    assert!(text.contains("REJOUER"), "{text}");
    assert!(
        text.contains("rename-modules"),
        "names the failing pass: {text}"
    );
    assert_eq!(
        marker(&env.raw, pid).await,
        None,
        "marker removed after the failure"
    );

    // The cause is gone (constraint dropped): the replay finishes and marks.
    raw_run(
        &env.raw,
        "MATCH (m:Module {id: 'squat-' + $p}) DETACH DELETE m",
        &p,
        &link,
        &real,
    )
    .await;
    assert_eq!(
        path_migration::run(&env.client, &slug, true, false)
            .await
            .unwrap(),
        0
    );
    assert_eq!(marker(&env.raw, pid).await.as_deref(), Some("true"));
    assert_eq!(
        strings(
            &env.raw,
            "MATCH (m:Module {project_id: $p}) RETURN m.stable_key AS v",
            &p
        )
        .await,
        vec![format!("{link}/src/m.rs::m")]
    );
    cleanup(&env.raw, pid).await;
}

async fn raw_run(raw: &Graph, cypher: &str, p: &str, link: &str, real: &str) {
    raw.run(
        query(cypher)
            .param("p", p)
            .param("link", link)
            .param("real", real),
    )
    .await
    .unwrap();
}

/// `update_project` with a new root rewrites BOTH normalized roots: the file
/// identities are computed against them. A name-only update leaves them alone,
/// and an update with nothing to set writes nothing.
#[tokio::test]
async fn updating_the_root_rewrites_the_declared_and_canonical_paths() {
    let Some(env) = env().await else {
        return;
    };
    let (real, link) = linked_repo("upd");
    let slug = format!("upd-{}", Uuid::new_v4());
    // No root yet: the boot-time repair (other tests run it concurrently) skips a
    // project without one, so it cannot write an older value over the update.
    let pid = Uuid::new_v4();
    let p = pid.to_string();
    env.raw
        .run(
            query("CREATE (:Project {id: $id, name: $slug, slug: $slug, created_at: datetime(), watch_enabled: false, profile: 'software'})")
                .param("id", p.clone())
                .param("slug", slug.clone()),
        )
        .await
        .unwrap();

    env.client
        .update_project(pid, None, None, Some(format!("{link}/")), None)
        .await
        .unwrap();
    assert_eq!(
        opt_string(
            &env.raw,
            "MATCH (pr:Project {id: $p}) RETURN pr.declared_path AS v",
            &p
        )
        .await,
        Some(link.clone()),
        "declared: no trailing slash, the link kept"
    );
    assert_eq!(
        opt_string(
            &env.raw,
            "MATCH (pr:Project {id: $p}) RETURN pr.canonical_path AS v",
            &p
        )
        .await,
        Some(real),
        "canonical: the link resolved"
    );

    // Name only: the roots are untouched.
    env.client
        .update_project(pid, Some("renamed".into()), None, None, None)
        .await
        .unwrap();
    assert_eq!(
        opt_string(
            &env.raw,
            "MATCH (pr:Project {id: $p}) RETURN pr.declared_path AS v",
            &p
        )
        .await,
        Some(link)
    );
    // Nothing to set: Ok, no write.
    env.client
        .update_project(pid, None, None, None, None)
        .await
        .unwrap();
    cleanup(&env.raw, pid).await;
}

/// A project that does not exist has no vocabulary: paths are kept as given.
#[tokio::test]
async fn an_unknown_project_has_no_path_policy() {
    let Some(env) = env().await else {
        return;
    };
    let policy = env.client.path_policy(Uuid::new_v4()).await.unwrap();
    assert!(!policy.declared);
    assert_eq!(policy.stored("/a/b.rs"), "/a/b.rs");
}

/// The root repair skips a `Project` node that has no readable id instead of
/// failing the whole boot pass.
#[tokio::test]
async fn the_root_repair_skips_a_project_without_an_id() {
    let Some(env) = env().await else {
        return;
    };
    let tag = format!("/idless-{}", Uuid::new_v4());
    env.raw
        .run(
            query("CREATE (:Project {name: 'idless', slug: $tag, root_path: $tag})")
                .param("tag", tag.clone()),
        )
        .await
        .unwrap();
    let result = env.client.ensure_project_canonical_paths().await;
    env.raw
        .run(query("MATCH (p:Project {slug: $tag}) DELETE p").param("tag", tag))
        .await
        .unwrap();
    let (scanned, _updated) = result.unwrap();
    assert!(scanned >= 1);
}

/// The real binary: `orchestrator migrate file-paths` reaches the migration,
/// prints its measures, and exits with the migration's own code — 0 for a
/// dry-run, 2 for a purge the gate refuses, non-zero for an unknown project.
/// (The subprocess inherits the connection through its environment only.)
#[tokio::test]
async fn the_cli_dispatches_the_migration_and_forwards_its_exit_code() {
    let Some(env) = env().await else {
        return;
    };
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:17688".into());
    let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into());
    let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testpassword".into());
    let orchestrator = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_orchestrator"))
            .args(["migrate", "file-paths"])
            .args(args)
            .env("NEO4J_URI", &uri)
            .env("NEO4J_USER", &user)
            .env("NEO4J_PASSWORD", &password)
            .env("HOME", sandbox_home())
            .output()
            .unwrap()
    };

    let (slug, pid, real, link) = alias_project(&env, "cli").await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/src/a.rs"),
        &format!("{pid}:src/a.rs"),
    )
    .await;
    put_file(
        &env.raw,
        pid,
        &format!("{real}/.wl/br/src/lonely.rs"),
        &format!("{pid}:lonely"),
    )
    .await;
    put_note(
        &env.raw,
        pid,
        "lonely",
        &[&format!("{real}/.wl/br/src/lonely.rs")],
    )
    .await;

    // Dry-run: exit 0, nothing written.
    let out = orchestrator(&["--project", &slug]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("Dry-run"));
    assert_eq!(marker(&env.raw, pid).await, None);

    // --purge-wl without --apply is refused by the argument parser.
    let out = orchestrator(&["--project", &slug, "--purge-wl"]);
    assert_eq!(out.status.code(), Some(2), "clap usage error");
    assert_eq!(marker(&env.raw, pid).await, None);

    // Apply + purge: the gate refuses (a note would lose its last anchor) -> 2.
    let out = orchestrator(&["--project", &slug, "--apply", "--purge-wl"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("Purge .wl REFUSÉE"));
    assert_eq!(
        paths_of_files(&env.raw, pid).await,
        vec![
            format!("{real}/.wl/br/src/lonely.rs"),
            format!("{link}/src/a.rs")
        ],
        "renamed, nothing purged"
    );

    // Unknown project: an error, a non-zero exit.
    let out = orchestrator(&["--project", &format!("nope-{}", Uuid::new_v4())]);
    assert_eq!(out.status.code(), Some(1));
    cleanup(&env.raw, pid).await;
}

/// The same two reads through the store trait objects the server holds: the
/// trait forwards to the Neo4j implementation, it does not answer for it.
#[tokio::test]
async fn the_store_traits_forward_the_path_reads_to_neo4j() {
    use project_orchestrator::neo4j::traits::{AnalyticsStore, ProjectStore};
    let Some(env) = env().await else {
        return;
    };
    let (_slug, pid, real, link) = alias_project(&env, "traits").await;
    env.client.mark_paths_declared(pid).await.unwrap();
    let projects: &dyn ProjectStore = &env.client;
    let policy = projects.path_policy(pid).await.unwrap();
    assert!(policy.declared);
    assert_eq!(
        policy.stored(&format!("{real}/src/a.rs")),
        format!("{link}/src/a.rs")
    );
    let analytics: &dyn AnalyticsStore = &env.client;
    let (scanned, _updated) = analytics.ensure_project_canonical_paths().await.unwrap();
    assert!(scanned >= 1);
    cleanup(&env.raw, pid).await;
}
