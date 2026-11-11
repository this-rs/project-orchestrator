//! The obrain storage backend behind the factory (`src/storage.rs`,
//! `src/obrain/`).
//!
//! What the cross-backend conformance suite (`tests/store_conformance.rs`,
//! `obrain_project::*`) cannot say, because it is specific to this backend:
//! how it is selected and configured, that it survives a close and a reopen,
//! that it is the single writer of its base, and that everything it has not
//! ported refuses by name.
//!
//! Every base opened here lives in a temporary directory created by the test.
//! No test reads a configured location: the configuration itself is a YAML
//! file written in that same temporary directory.

use project_orchestrator::storage::StorageBackend;
use project_orchestrator::Config;
use std::path::Path;

/// A configuration loaded the way the server loads it — from a YAML file with
/// its NESTED keys — pointing at nothing outside `dir`.
fn config_from_yaml(dir: &Path, storage_section: &str) -> Config {
    let file = dir.join("config.yaml");
    std::fs::write(&file, format!("setup_completed: true\n{storage_section}")).unwrap();
    Config::from_yaml_and_env(Some(&file)).unwrap()
}

/// The error of a factory that must refuse.
async fn refusal(config: &Config) -> String {
    match StorageBackend::from_config(config).await {
        Ok(_) => panic!("the factory must refuse this configuration"),
        Err(e) => format!("{e:#}"),
    }
}

#[tokio::test]
async fn an_unknown_backend_names_itself_and_the_three_accepted_values() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_from_yaml(dir.path(), "storage:\n  backend: mystery\n");
    if config.storage_backend != "mystery" {
        // STORAGE_BACKEND is set in this shell and wins over the file.
        eprintln!("skipped: STORAGE_BACKEND overrides the YAML of this test");
        return;
    }
    let msg = refusal(&config).await;
    for expected in ["mystery", "'default'", "'memory'", "'obrain'"] {
        assert!(msg.contains(expected), "« {expected} » missing from: {msg}");
    }
}

#[cfg(not(feature = "obrain-backend"))]
#[tokio::test]
async fn without_the_feature_obrain_is_refused_at_startup_and_says_why() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    let config = config_from_yaml(
        dir.path(),
        &format!(
            "storage:\n  backend: obrain\n  obrain_path: \"{}\"\n",
            base.display()
        ),
    );
    if config.storage_backend != "obrain" {
        eprintln!("skipped: STORAGE_BACKEND overrides the YAML of this test");
        return;
    }
    let msg = refusal(&config).await;
    assert!(
        msg.contains("obrain") && msg.contains("obrain-backend"),
        "the refusal names the backend and the feature to build with: {msg}"
    );
    assert!(
        !msg.contains("unknown storage backend"),
        "`obrain` is a known value, not a typo: {msg}"
    );
    assert!(!base.exists(), "nothing was created at the configured path");
}

#[cfg(feature = "obrain-backend")]
mod with_the_feature {
    use super::*;
    use chrono::Utc;
    use obrain_common::types::Value;
    use project_orchestrator::meilisearch::SearchStore;
    use project_orchestrator::neo4j::models::{ProgressKind, ProjectNode, ProjectProfile};
    use project_orchestrator::neo4j::traits::*;
    use project_orchestrator::obrain::{
        defauts_conserves, methodes_portees, methodes_refusees, NonPorte, ObrainGraphStore,
        ObrainSearchStore, ObrainTrajectoryStore,
    };
    use uuid::Uuid;

    fn a_project(name: &str) -> ProjectNode {
        let id = Uuid::new_v4();
        ProjectNode {
            id,
            name: name.to_string(),
            slug: format!("obrain-{id}"),
            root_path: String::new(),
            description: Some(format!("description of {name}")),
            created_at: Utc::now(),
            last_synced: None,
            analytics_computed_at: None,
            last_co_change_computed_at: None,
            default_note_energy: None,
            scaffolding_override: None,
            sharing_policy: None,
            watch_enabled: false,
            profile: ProjectProfile::Software,
        }
    }

    fn fields(p: &ProjectNode) -> serde_json::Value {
        serde_json::to_value(p).unwrap()
    }

    async fn open(path: &Path) -> ObrainGraphStore {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || ObrainGraphStore::open(path))
            .await
            .unwrap()
            .unwrap()
    }

    /// The `NonPorte` at the root of an error, or a panic naming what came.
    fn not_ported<T: std::fmt::Debug>(result: anyhow::Result<T>) -> NonPorte {
        let err = result.expect_err("a method that is not ported must refuse");
        *err.downcast_ref::<NonPorte>()
            .unwrap_or_else(|| panic!("expected a NonPorte, got: {err:#}"))
    }

    // ------------------------------------------------------------------
    // Selection and configuration
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn the_factory_builds_obrain_from_the_nested_yaml_keys() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let config = config_from_yaml(
            dir.path(),
            &format!(
                "storage:\n  backend: obrain\n  obrain_path: \"{}\"\n",
                base.display()
            ),
        );
        if config.storage_backend != "obrain"
            || config.storage_obrain_path.as_deref() != Some(base.to_str().unwrap())
        {
            // An environment override would send the factory elsewhere than
            // this temporary directory: never follow it.
            eprintln!("skipped: STORAGE_BACKEND / STORAGE_OBRAIN_PATH override the YAML");
            return;
        }

        // Neither Neo4j nor Meilisearch is configured to anything reachable
        // from this test: the obrain backend must not need them.
        let backend = StorageBackend::from_config(&config).await.unwrap();
        assert!(base.is_dir(), "the base was created at the configured path");

        // The graph is the obrain store: the ported domain answers…
        let p = a_project("through the factory");
        backend.graph.create_project(&p).await.unwrap();
        assert_eq!(
            fields(&backend.graph.get_project(p.id).await.unwrap().unwrap()),
            fields(&p)
        );
        // …and the rest of the storage refuses by name, graph, search and
        // trajectories alike.
        assert_eq!(
            not_ported(backend.graph.list_workspaces().await),
            NonPorte("WorkspaceStore::list_workspaces")
        );
        assert_eq!(
            not_ported(backend.search.delete_code("x").await),
            NonPorte("SearchStore::delete_code")
        );
        let trajectories = backend.trajectories.count().await.unwrap_err();
        assert!(
            trajectories
                .to_string()
                .contains("non porté sur le backend obrain : TrajectoryStore::count"),
            "{trajectories}"
        );
        // Nothing is claimed that no lot has proven.
        assert_eq!(backend.capabilities, StoreCapabilities::default());
    }

    #[tokio::test]
    async fn obrain_without_a_path_is_refused_and_names_both_keys() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_from_yaml(dir.path(), "storage:\n  backend: obrain\n");
        if config.storage_backend != "obrain" || config.storage_obrain_path.is_some() {
            eprintln!("skipped: STORAGE_BACKEND / STORAGE_OBRAIN_PATH override the YAML");
            return;
        }
        let msg = refusal(&config).await;
        assert!(
            msg.contains("storage.obrain_path") && msg.contains("STORAGE_OBRAIN_PATH"),
            "the refusal names the YAML key and the variable: {msg}"
        );
    }

    // ------------------------------------------------------------------
    // Persistence: close, reopen, same values, same order
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn projects_survive_a_close_and_a_reopen_with_values_and_order() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");

        let store = open(&base).await;
        // Two share a name: the order between them is the tie-break.
        let mut projects = [
            a_project("persist b"),
            a_project("persist tie"),
            a_project("persist a"),
            a_project("persist tie"),
        ];
        let deleted = a_project("persist deleted");
        for p in projects.iter().chain([&deleted]) {
            store.create_project(p).await.unwrap();
        }
        // An update, a cleared description, both timestamps, a delete: every
        // kind of write must be on disk, not only the creation.
        store
            .update_project(
                projects[0].id,
                Some("persist renamed".to_string()),
                Some(None),
                Some("/nonexistent/obrain-persist".to_string()),
                Some(ProjectProfile::Work),
            )
            .await
            .unwrap();
        store.update_project_synced(projects[1].id).await.unwrap();
        store
            .update_project_analytics_timestamp(projects[1].id)
            .await
            .unwrap();
        store
            .delete_project(deleted.id, &deleted.name)
            .await
            .unwrap();

        let before = store.list_projects().await.unwrap();
        assert_eq!(before.len(), 4);
        let policy_before = store.path_policy(projects[0].id).await.unwrap();
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        let after = store.list_projects().await.unwrap();
        assert_eq!(
            after.iter().map(fields).collect::<Vec<_>>(),
            before.iter().map(fields).collect::<Vec<_>>(),
            "same values, same order, after the reopen"
        );
        // The order is the stated one, not an accident of the first run.
        projects[0].name = "persist renamed".to_string();
        projects.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.id.to_string().cmp(&b.id.to_string()))
        });
        assert_eq!(
            after.iter().map(|p| p.id).collect::<Vec<_>>(),
            projects.iter().map(|p| p.id).collect::<Vec<_>>()
        );

        // The indexes were rebuilt from the base: by id, by slug.
        for p in &before {
            assert_eq!(
                fields(&store.get_project(p.id).await.unwrap().unwrap()),
                fields(p)
            );
            assert_eq!(
                store
                    .get_project_by_slug(&p.slug)
                    .await
                    .unwrap()
                    .map(|f| f.id),
                Some(p.id)
            );
        }
        let renamed = store
            .get_project(projects_id(&before, "persist renamed"))
            .await;
        let renamed = renamed.unwrap().unwrap();
        assert_eq!(renamed.description, Some(String::new()));
        assert_eq!(renamed.root_path, "/nonexistent/obrain-persist");
        assert_eq!(renamed.profile, ProjectProfile::Work);
        assert_eq!(store.path_policy(renamed.id).await.unwrap(), policy_before);
        let stamped = before.iter().find(|p| p.last_synced.is_some()).unwrap();
        assert!(stamped.analytics_computed_at.is_some());

        // What was deleted stays deleted, and its slug is free.
        assert!(store.get_project(deleted.id).await.unwrap().is_none());
        assert!(store
            .get_project_by_slug(&deleted.slug)
            .await
            .unwrap()
            .is_none());
        // Uniqueness holds against what was written before the reopen.
        let mut same_slug = a_project("persist intruder");
        same_slug.slug = before[0].slug.clone();
        assert!(store.create_project(&same_slug).await.is_err());
        let mut same_id = a_project("persist intruder");
        same_id.id = before[0].id;
        assert!(store.create_project(&same_id).await.is_err());
        assert_eq!(store.list_projects().await.unwrap().len(), 4);
    }

    fn projects_id(projects: &[ProjectNode], name: &str) -> Uuid {
        projects.iter().find(|p| p.name == name).unwrap().id
    }

    /// A write does not wait for a clean close: the files as they are right
    /// after the acknowledgement, copied while the base is still open, reopen
    /// with the write in them. This is the PROCESS-crash model (what the OS
    /// holds survives); it says nothing of a power cut. Measured: the test
    /// passes just the same with `commit_wal()` removed from the adapter, so
    /// it does not prove the fsync — only that nothing waits for `close`.
    #[tokio::test]
    async fn an_acknowledged_write_survives_without_a_clean_close() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let copy = dir.path().join("copy");

        let store = open(&base).await;
        let p = a_project("durable");
        store.create_project(&p).await.unwrap();
        store.update_project_synced(p.id).await.unwrap();
        let written = store.get_project(p.id).await.unwrap().unwrap();

        // The files as they are on disk NOW, while the base is still open and
        // nothing has closed it: what a crash at this instant would leave.
        copy_dir(&base, &copy);

        let reopened = open(&copy).await;
        assert_eq!(
            fields(&reopened.get_project(p.id).await.unwrap().unwrap()),
            fields(&written)
        );
        drop(store);
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    // ------------------------------------------------------------------
    // Single writer
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn a_base_already_open_for_writing_cannot_be_opened_again() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let first = open(&base).await;

        let again = base.clone();
        let second = tokio::task::spawn_blocking(move || ObrainGraphStore::open(again))
            .await
            .unwrap();
        assert!(
            second.is_err(),
            "a second writable handle on the same base must be refused"
        );

        // The first handle is intact, and once it is closed the base opens.
        let p = a_project("single writer");
        first.create_project(&p).await.unwrap();
        first.close().await.unwrap();
        drop(first);
        let reopened = open(&base).await;
        assert!(reopened.get_project(p.id).await.unwrap().is_some());
    }

    // ------------------------------------------------------------------
    // A base this backend did not write (what a migration will produce)
    // ------------------------------------------------------------------

    /// A `Project` node written straight through obrain, with a relationship
    /// to a node of a domain this backend has not ported.
    fn write_foreign_base(base: &Path, id: Uuid, slug: &str) {
        let db = obrain_engine::ObrainDB::open(base).unwrap();
        let project = db.create_node_with_props(
            &["Project"],
            vec![
                ("id", Value::from(id.to_string())),
                ("name", Value::from("migrated")),
                ("slug", Value::from(slug)),
                (
                    "created_at",
                    Value::from("2026-01-02T03:04:05.123456789+00:00"),
                ),
                ("default_note_energy", Value::from(0.25)),
                ("scaffolding_override", Value::from(3_i64)),
                ("last_synced", Value::from("2026-02-03T04:05:06+00:00")),
            ],
        );
        let plan = db.create_node_with_props(&["Plan"], vec![("id", Value::from("p1"))]);
        db.create_edge(project, plan, "HAS_PLAN");
        db.commit_wal().unwrap();
        db.close().unwrap();
    }

    #[tokio::test]
    async fn a_project_written_outside_is_read_and_its_cascade_delete_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let id = Uuid::new_v4();
        let slug = format!("migrated-{id}");
        {
            let (base, slug) = (base.clone(), slug.clone());
            tokio::task::spawn_blocking(move || write_foreign_base(&base, id, &slug))
                .await
                .unwrap();
        }

        let store = open(&base).await;
        // Read like the Neo4j adapter reads a node: absent properties take
        // their documented defaults, the setter-only fields are read.
        let p = store
            .get_project(id)
            .await
            .unwrap()
            .expect("indexed at open");
        assert_eq!(p.name, "migrated");
        assert_eq!(p.description, None, "no property, no description");
        assert_eq!(p.root_path, "");
        assert!(p.watch_enabled, "absent means watched");
        assert_eq!(p.profile, ProjectProfile::Software);
        assert_eq!(p.default_note_energy, Some(0.25));
        assert_eq!(p.scaffolding_override, Some(3));
        assert_eq!(
            p.created_at.to_rfc3339(),
            "2026-01-02T03:04:05.123456789+00:00"
        );
        assert!(p.last_synced.is_some());
        assert_eq!(
            store
                .get_project_by_slug(&slug)
                .await
                .unwrap()
                .map(|p| p.id),
            Some(id)
        );

        // It carries a relationship this backend cannot cascade: refused by
        // name, and nothing is deleted.
        let err = store.delete_project(id, "migrated").await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<NonPorte>(),
            Some(&NonPorte("ProjectStore::delete_project (cascade)")),
            "{err:#}"
        );
        assert!(format!("{err:#}").contains("1 relation"), "{err:#}");
        assert!(store.get_project(id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_project_node_without_an_id_fails_the_open_instead_of_being_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        {
            let base = base.clone();
            tokio::task::spawn_blocking(move || {
                let db = obrain_engine::ObrainDB::open(&base).unwrap();
                db.create_node_with_props(&["Project"], vec![("name", Value::from("orphan"))]);
                db.commit_wal().unwrap();
                db.close().unwrap();
            })
            .await
            .unwrap();
        }
        let err = tokio::task::spawn_blocking(move || ObrainGraphStore::open(base))
            .await
            .unwrap()
            .err()
            .expect("an unaddressable Project node must fail the open");
        assert!(
            format!("{err:#}").contains("sans propriété `id`"),
            "{err:#}"
        );
    }

    // ------------------------------------------------------------------
    // The rest of the contract refuses by name
    // ------------------------------------------------------------------

    /// One method per kind of signature and per corner of the contract: first
    /// and last trait of `GraphStore`, a method the contract defaults to a
    /// silent `Ok`, the search store, and the default of `GraphStore` itself,
    /// which reaches a refused method.
    #[tokio::test]
    async fn what_is_not_ported_refuses_by_name_and_never_answers_empty() {
        let store = tokio::task::spawn_blocking(ObrainGraphStore::open_temporary)
            .await
            .unwrap()
            .unwrap();
        let id = Uuid::new_v4();

        assert_eq!(
            not_ported(store.get_workspace(id).await),
            NonPorte("WorkspaceStore::get_workspace")
        );
        assert_eq!(
            not_ported(store.get_plan_tasks(id).await),
            NonPorte("PlanStore::get_plan_tasks")
        );
        assert_eq!(
            not_ported(store.list_project_files(id).await),
            NonPorte("CodeStore::list_project_files")
        );
        assert_eq!(
            not_ported(store.delete_note(id).await),
            NonPorte("NoteStore::delete_note")
        );
        // Defaulted by the contract to `Ok((0, 0))`: refused all the same.
        assert_eq!(
            not_ported(store.ensure_project_canonical_paths().await),
            NonPorte("AnalyticsStore::ensure_project_canonical_paths")
        );
        // The default of `GraphStore` loops over a refused getter: with an id
        // to look up it refuses, by the name of that getter.
        assert_eq!(
            not_ported(store.get_progress_batch(ProgressKind::Project, &[id]).await),
            NonPorte("ReleaseStore::get_project_tasks")
        );
        assert_eq!(
            not_ported(ObrainSearchStore.delete_code("x").await),
            NonPorte("SearchStore::delete_code")
        );
        let _ = ObrainTrajectoryStore;

        // The message is the stated one.
        let msg = store.list_workspaces().await.unwrap_err().to_string();
        assert_eq!(
            msg,
            "non porté sur le backend obrain : WorkspaceStore::list_workspaces"
        );
    }

    /// The generator skips nothing: every `async fn` of the two contracts is
    /// either refused, or in a trait implemented by hand, or one of the few
    /// that cannot refuse and keep the contract's default.
    #[test]
    fn every_method_of_the_contracts_is_accounted_for() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let async_fns = |file: &str| {
            let source = std::fs::read_to_string(root.join(file)).unwrap();
            // The traits end where the unit tests of the file begin.
            let contract = source.split("#[cfg(test)]").next().unwrap();
            contract
                .lines()
                .filter(|l| l.trim_start().starts_with("async fn "))
                .count()
        };
        let in_contracts =
            async_fns("src/neo4j/traits.rs") + async_fns("src/meilisearch/traits.rs");

        let refused = methodes_refusees();
        let ported = methodes_portees();
        let kept = defauts_conserves();
        // `GraphStore::capabilities` is the one method that is not `async`.
        let ported_async = ported
            .iter()
            .filter(|m| **m != "GraphStore::capabilities")
            .count();
        assert_eq!(
            refused.len() + ported_async + kept.len(),
            in_contracts,
            "refused {} + ported {} + kept {} must be the {} async fn of the contracts",
            refused.len(),
            ported_async,
            kept.len(),
            in_contracts
        );

        // This lot: exactly ProjectStore (and GraphStore's own two methods).
        let mut domains: Vec<&str> = ported
            .iter()
            .map(|m| m.split("::").next().unwrap())
            .collect();
        domains.dedup();
        assert_eq!(domains, ["GraphStore", "ProjectStore"]);
        assert_eq!(
            ported
                .iter()
                .filter(|m| m.starts_with("ProjectStore::"))
                .count(),
            9
        );
        // What keeps a default is known and justified where it is declared.
        assert_eq!(kept, ["AnalyticsStore::run_data_migrations"]);
        // No name twice, no ported method refused.
        let mut all: Vec<&str> = refused.iter().chain(ported).chain(kept).copied().collect();
        let total = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), total, "a method is listed twice");
        eprintln!(
            "obrain backend: {} methods refused by name, {} ported, {} keeping the contract's default",
            refused.len(),
            ported.len(),
            kept.len()
        );
    }
}
