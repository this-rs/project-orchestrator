//! The obrain storage backend behind the factory (`src/storage.rs`,
//! `src/obrain/`).
//!
//! What the cross-backend conformance suite (`tests/store_conformance.rs`,
//! `obrain_project::*`) cannot say, because it is specific to this backend:
//! how it is selected and configured, that it survives a close and a reopen,
//! that a crash in the middle of a write leaves a base that opens, that it is
//! the single writer of its base, that readers do not wait behind an fsync, and
//! that everything it has not ported refuses by name.
//!
//! Every base opened here lives in a temporary directory created by the test.
//! No test reads a configured location: the configuration itself is a YAML
//! file written in that same temporary directory, and a test that would be
//! overridden by the environment FAILS naming the variable (it never passes
//! without having checked).

use project_orchestrator::storage::StorageBackend;
use project_orchestrator::Config;
use std::path::Path;

/// The variables that would override the YAML of these tests. A test that
/// loads a configuration refuses to run if one is set: passing silently would
/// be a test that verified nothing.
fn env_propre() {
    for v in [
        "STORAGE_BACKEND",
        "STORAGE_OBRAIN_PATH",
        "STORAGE_OBRAIN_CREATE",
    ] {
        assert!(
            std::env::var_os(v).is_none(),
            "{v} is set in the environment of this test and would override the YAML it loads: \
             run it with `env -u {v} cargo test …`"
        );
    }
}

/// A configuration loaded the way the server loads it — from a YAML file with
/// its NESTED keys — pointing at nothing outside `dir`.
fn config_from_yaml(dir: &Path, storage_section: &str) -> Config {
    env_propre();
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
    assert_eq!(config.storage_backend, "mystery");
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
            "storage:\n  backend: obrain\n  obrain_path: \"{}\"\n  obrain_create: true\n",
            base.display()
        ),
    );
    assert_eq!(config.storage_backend, "obrain");
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
    use project_orchestrator::neo4j::models::{
        CommitNode, FileChangedInfo, PlanNode, PlanStatus, ProgressKind, ProjectNode,
        ProjectProfile,
    };
    use project_orchestrator::neo4j::traits::*;
    use project_orchestrator::obrain::{
        defauts_conserves, methodes_portees, methodes_qui_servent, methodes_refusees,
        refusees_chat, refusees_commit, refusees_decision, refusees_featuregraph, refusees_note,
        refusees_persona, refusees_plan, refusees_planrun, refusees_protocolrun, refusees_release,
        refusees_sharing, refusees_skill, Creation, Durabilite, Interruption, NonPorte,
        ObrainGraphStore, ObrainSearchStore, RapportOuverture, MARQUEUR_PO, REFUS_ECRITS_A_LA_MAIN,
        SCHEMA_PO,
    };
    use std::sync::Arc;
    use std::time::{Duration, Instant};
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
        tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
            .await
            .unwrap()
            .unwrap()
    }

    async fn try_open(path: &Path, creation: Creation) -> anyhow::Result<ObrainGraphStore> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, creation))
            .await
            .unwrap()
    }

    async fn temporary(durabilite: Durabilite) -> ObrainGraphStore {
        tokio::task::spawn_blocking(move || ObrainGraphStore::open_temporary_avec(durabilite))
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

    /// Write straight into a base through the engine, the way another writer
    /// (a migration, a crash) would leave it: `f` runs, the WAL is committed,
    /// the base is closed.
    async fn write_foreign(base: &Path, f: impl FnOnce(&obrain_engine::ObrainDB) + Send + 'static) {
        let base = base.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let db = obrain_engine::ObrainDB::open(&base).unwrap();
            f(&db);
            db.commit_wal().unwrap();
            db.close().unwrap();
            // What the migration tool does once its base is written: say it is PO's.
            std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
        })
        .await
        .unwrap();
    }

    /// The `created_at` of every `SUPERSEDES` edge leaving a `Decision` node, as the
    /// ENGINE reads them in a closed base (one entry per edge).
    async fn supersedes_on_disk(base: &Path) -> Vec<Option<String>> {
        let base = base.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let db = obrain_engine::ObrainDB::open(&base).unwrap();
            let mut out = Vec::new();
            for n in db.store().nodes_by_label("Decision") {
                for (_, a) in db
                    .store()
                    .edges_from(n, obrain_core::graph::Direction::Outgoing)
                {
                    let e = db.get_edge(a).unwrap();
                    if &*e.edge_type == "SUPERSEDES" {
                        out.push(
                            e.get_property("created_at")
                                .and_then(|v| v.as_str().map(str::to_string)),
                        );
                    }
                }
            }
            db.close().unwrap();
            out
        })
        .await
        .unwrap()
    }

    /// A decision with the given description, no project, proposed.
    fn a_decision(description: &str) -> project_orchestrator::neo4j::models::DecisionNode {
        project_orchestrator::neo4j::models::DecisionNode {
            id: Uuid::new_v4(),
            description: description.to_string(),
            rationale: "parce que".to_string(),
            alternatives: vec![],
            chosen_option: None,
            decided_by: "alice".to_string(),
            decided_at: Utc::now(),
            status: project_orchestrator::neo4j::models::DecisionStatus::Proposed,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        }
    }

    // ------------------------------------------------------------------
    // Selection, configuration, and the guards of opening
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn the_factory_builds_obrain_from_the_nested_yaml_keys() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let config = config_from_yaml(
            dir.path(),
            &format!(
                "storage:\n  backend: obrain\n  obrain_path: \"{}\"\n  obrain_create: true\n",
                base.display()
            ),
        );
        assert_eq!(config.storage_backend, "obrain");
        assert_eq!(
            config.storage_obrain_path.as_deref(),
            Some(base.to_str().unwrap())
        );
        assert!(config.storage_obrain_create);

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
            not_ported(backend.graph.compute_coupling_matrix(Uuid::new_v4()).await),
            NonPorte("WorkspaceStore::compute_coupling_matrix")
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
        assert_eq!(config.storage_backend, "obrain");
        assert!(config.storage_obrain_path.is_none());
        let msg = refusal(&config).await;
        assert!(
            msg.contains("storage.obrain_path") && msg.contains("STORAGE_OBRAIN_PATH"),
            "the refusal names the YAML key and the variable: {msg}"
        );
    }

    /// A relative path depends on where the process happens to start.
    #[tokio::test]
    async fn a_relative_path_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_from_yaml(
            dir.path(),
            "storage:\n  backend: obrain\n  obrain_path: \"data/obrain\"\n  obrain_create: true\n",
        );
        let msg = refusal(&config).await;
        assert!(
            msg.contains("relatif") && msg.contains("absolu"),
            "names the problem and the fix: {msg}"
        );
        assert!(
            !Path::new("data/obrain").exists(),
            "nothing was created relative to the working directory"
        );
    }

    /// A typo in the path would otherwise be a brand-new empty base that PO
    /// then serves, with zero projects, as if nothing were wrong.
    #[tokio::test]
    async fn a_missing_or_empty_directory_is_not_a_base_unless_creation_is_asked() {
        let dir = tempfile::tempdir().unwrap();
        let config = |path: &Path, create: bool| {
            config_from_yaml(
                dir.path(),
                &format!(
                    "storage:\n  backend: obrain\n  obrain_path: \"{}\"\n  obrain_create: {create}\n",
                    path.display()
                ),
            )
        };

        let typo = dir.path().join("obrian");
        let msg = refusal(&config(&typo, false)).await;
        assert!(
            msg.contains("storage.obrain_create") && msg.contains("répertoire absent"),
            "{msg}"
        );
        assert!(!typo.exists(), "a refused open creates nothing");

        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        let msg = refusal(&config(&empty, false)).await;
        assert!(msg.contains("répertoire vide"), "{msg}");
        assert_eq!(std::fs::read_dir(&empty).unwrap().count(), 0);

        // Creation is a decision, and then it works — once; the next start
        // needs no flag (the base exists).
        let created = dir.path().join("created");
        StorageBackend::from_config(&config(&created, true))
            .await
            .unwrap();
        assert!(created.join("substrate.meta").is_file());
        // The first start's store is dropped, so the base is free again.
        StorageBackend::from_config(&config(&created, false))
            .await
            .expect("an existing base opens without the creation flag");
    }

    /// The engine writes its own files into a directory it is given; over a
    /// directory that is something else, that is destruction.
    #[tokio::test]
    async fn a_directory_that_is_not_a_base_is_never_written_into() {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("somebody-elses-data");
        std::fs::create_dir(&other).unwrap();
        let precious: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        // The name of an engine file, without the `substrate.meta` that makes a
        // directory a base: the case where the engine would create over it.
        std::fs::write(other.join("substrate.nodes"), &precious).unwrap();
        std::fs::write(other.join("notes.txt"), b"keep me").unwrap();

        for creation in [Creation::Interdite, Creation::Permise] {
            let err = try_open(&other, creation).await.err().unwrap();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("n'est pas vide") && msg.contains("Rien n'a été touché"),
                "{creation:?}: {msg}"
            );
        }
        assert_eq!(
            std::fs::read(other.join("substrate.nodes")).unwrap(),
            precious
        );
        assert_eq!(std::fs::read(other.join("notes.txt")).unwrap(), b"keep me");
        assert_eq!(std::fs::read_dir(&other).unwrap().count(), 2);

        // A plain file is not a base either.
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"x").unwrap();
        let msg = format!(
            "{:#}",
            try_open(&file, Creation::Permise).await.err().unwrap()
        );
        assert!(msg.contains("n'est pas un répertoire"), "{msg}");
    }

    /// Any directory with a `substrate.meta` is an obrain base; only one that
    /// carries PO's marker is PO's. A base created by PO carries it from its first
    /// instant; an obrain base of another tool is refused, and left untouched.
    #[tokio::test]
    async fn an_obrain_base_that_is_not_po_s_is_refused_and_a_created_one_is_marked() {
        let dir = tempfile::tempdir().unwrap();
        let ours = dir.path().join("ours");
        open(&ours).await.close().await.unwrap();
        assert_eq!(
            std::fs::read_to_string(ours.join(MARQUEUR_PO)).unwrap(),
            SCHEMA_PO,
            "a base PO creates carries the marker"
        );

        // A raw obrain base, written by something else, without the marker.
        let theirs = dir.path().join("theirs");
        {
            let theirs = theirs.clone();
            tokio::task::spawn_blocking(move || {
                let db = obrain_engine::ObrainDB::open(&theirs).unwrap();
                db.create_node_with_props(&["Project"], vec![("id", Value::from("x"))]);
                db.commit_wal().unwrap();
                db.close().unwrap();
            })
            .await
            .unwrap();
        }
        assert!(theirs.join("substrate.meta").is_file());
        let before: Vec<_> = std::fs::read_dir(&theirs)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        for creation in [Creation::Interdite, Creation::Permise] {
            let msg = format!("{:#}", try_open(&theirs, creation).await.err().unwrap());
            assert!(
                msg.contains(MARQUEUR_PO) && msg.contains("pas une base PO"),
                "{creation:?}: {msg}"
            );
        }
        let after: Vec<_> = std::fs::read_dir(&theirs)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(after, before, "a refused base is not written into");
        assert!(!theirs.join(MARQUEUR_PO).exists());
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
        assert_eq!(
            *store.rapport_d_ouverture(),
            RapportOuverture::default(),
            "a clean close leaves nothing to repair"
        );
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
        let renamed = before
            .iter()
            .find(|p| p.name == "persist renamed")
            .unwrap()
            .id;
        let renamed = store.get_project(renamed).await.unwrap().unwrap();
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

    // ------------------------------------------------------------------
    // Durability: the fsync, proved by obrain's own oracle
    // ------------------------------------------------------------------

    /// obrain's oracle counts the fsyncs the WAL has REALLY run. Between two
    /// acknowledgements of the single writer it must have grown: every
    /// acknowledged write ran a fsync before it answered.
    #[tokio::test]
    async fn every_acknowledged_write_ran_a_wal_fsync() {
        let store = temporary(Durabilite::Wal).await;
        let last = std::cell::Cell::new(store.fsyncs_du_wal().expect("a persistent base"));
        let ran = |quoi: &str| {
            let n = store.fsyncs_du_wal().unwrap();
            assert!(
                n > last.get(),
                "{quoi}: no WAL fsync ran since the previous acknowledgement ({} → {n})",
                last.get()
            );
            last.set(n);
        };
        let mut ids = Vec::new();
        for i in 0..10 {
            let p = a_project(&format!("durable {i}"));
            store.create_project(&p).await.unwrap();
            ran("create_project");
            ids.push(p);
        }
        store
            .update_project(ids[0].id, Some("durable renamed".into()), None, None, None)
            .await
            .unwrap();
        ran("update_project");
        store.update_project_synced(ids[1].id).await.unwrap();
        ran("update_project_synced");
        store
            .update_project_analytics_timestamp(ids[2].id)
            .await
            .unwrap();
        ran("update_project_analytics_timestamp");
        // Delete is refused while other data lives in the base; here only projects do.
        store.delete_project(ids[3].id, "x").await.unwrap();
        ran("delete_project");
    }

    /// The same check, without the fsync: it must go RED. A durability test that
    /// cannot fail proves nothing (the first version of this file had one).
    #[tokio::test]
    async fn without_the_fsync_the_oracle_sees_none() {
        let store = temporary(Durabilite::Aucune).await;
        let before = store.fsyncs_du_wal().unwrap();
        for i in 0..10 {
            store
                .create_project(&a_project(&format!("not durable {i}")))
                .await
                .unwrap();
        }
        assert_eq!(
            store.fsyncs_du_wal().unwrap(),
            before,
            "with no durability step, no fsync runs — the check above would fail"
        );
    }

    /// The auth domain under the same oracle: every acknowledged auth write ran a
    /// WAL fsync before it answered (users, refresh tokens, MCP tokens, the
    /// rotation, the revocations and the purge).
    #[tokio::test]
    async fn every_acknowledged_auth_write_ran_a_wal_fsync() {
        let store = temporary(Durabilite::Wal).await;
        let last = std::cell::Cell::new(store.fsyncs_du_wal().expect("a persistent base"));
        let ran = |quoi: &str| {
            let n = store.fsyncs_du_wal().unwrap();
            assert!(
                n > last.get(),
                "{quoi}: no WAL fsync ran since the previous acknowledgement ({} → {n})",
                last.get()
            );
            last.set(n);
        };
        let u = store
            .create_password_user("durable@ex.test", "D", "h")
            .await
            .unwrap();
        ran("create_password_user");
        store
            .create_password_user("durable@ex.test", "D2", "h")
            .await
            .unwrap();
        ran("upsert_user (ON MATCH)");
        let futur = Utc::now() + chrono::Duration::hours(1);
        for k in 0..3 {
            store
                .create_refresh_token(u.id, &format!("durable-rt-{k}"), futur)
                .await
                .unwrap();
            ran("create_refresh_token");
        }
        store.consume_refresh_token("durable-rt-0").await.unwrap();
        ran("consume_refresh_token");
        store.revoke_refresh_token("durable-rt-1").await.unwrap();
        ran("revoke_refresh_token");
        store.revoke_all_user_tokens(u.id).await.unwrap();
        ran("revoke_all_user_tokens");
        store
            .create_mcp_token(u.id, "durable-jti", "l", "s", futur)
            .await
            .unwrap();
        ran("create_mcp_token");
        store.revoke_mcp_token(u.id, "durable-jti").await.unwrap();
        ran("revoke_mcp_token");
        store.purge_mcp_tokens(u.id).await.unwrap();
        ran("purge_mcp_tokens");
    }

    /// A value past the blob threshold is not recovered by the WAL: the write
    /// that carries it must run the full flush too (measured: lost after SIGKILL
    /// otherwise, see `SEUIL_COLONNE_BLOB`). Observed through the engine's own
    /// counter of fsyncs: a flush runs more of them than the WAL fsync alone.
    #[tokio::test]
    async fn a_write_past_the_blob_threshold_pays_the_full_flush() {
        let store = temporary(Durabilite::Wal).await;
        let fsyncs = || store.fsyncs_du_wal().unwrap();
        let mut small = a_project("small");
        small.description = Some("x".repeat(256));
        let t0 = fsyncs();
        store.create_project(&small).await.unwrap();
        let small_cost = fsyncs() - t0;
        let mut big = a_project("big");
        big.description = Some("x".repeat(257));
        let t1 = fsyncs();
        store.create_project(&big).await.unwrap();
        let big_cost = fsyncs() - t1;
        assert!(small_cost >= 1, "256 bytes: the WAL fsync ran");
        assert!(
            big_cost > small_cost,
            "257 bytes: the full flush too ({big_cost} fsyncs against {small_cost})"
        );
    }

    /// If the fsync fails, the write is applied in memory and the disk is in an
    /// unknown state. The store says so, and stops: nothing is read or written
    /// on a base whose durability is unknown, until a restart replays the log.
    #[tokio::test]
    async fn a_failed_fsync_makes_the_store_refuse_everything_and_says_why() {
        let store = temporary(Durabilite::EchoueAPourTest).await;
        let p = a_project("fsync fails");
        let err = store.create_project(&p).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("INDÉTERMINÉ") && msg.contains("durabilité"),
            "the error says the state is unknown: {msg}"
        );
        // Nothing is served, nothing more is accepted.
        for refus in [
            store.get_project(p.id).await.map(|_| ()),
            store.list_projects().await.map(|_| ()),
            store.create_project(&a_project("after")).await,
        ] {
            let msg = format!("{:#}", refus.unwrap_err());
            assert!(msg.contains("indéterminé"), "{msg}");
        }
    }

    // ------------------------------------------------------------------
    // Locks: readers do not wait behind an fsync
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn readers_do_not_wait_behind_a_slow_fsync() {
        // Every write waits 2 s for its fsync: the only timing this test relies on
        // is "a read takes less than that", with a wide margin for a loaded machine.
        let store = Arc::new(temporary(Durabilite::LenteAPourTest(2_000)).await);
        let p = a_project("read while a write fsyncs");
        let other = a_project("another project");
        store.create_project(&p).await.unwrap();
        store.create_project(&other).await.unwrap();

        // An UPDATE in flight (in-memory part done, fsync running) and a
        // CREATION in flight (fsync running before it is published).
        let writer = store.clone();
        let id = p.id;
        let update = tokio::spawn(async move {
            writer
                .update_project(id, Some("renamed under fsync".into()), None, None, None)
                .await
        });
        let creator = store.clone();
        let created = a_project("created under fsync");
        let creating = created.clone();
        let create = tokio::spawn(async move { creator.create_project(&creating).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !update.is_finished() && !create.is_finished(),
            "the writes must be in their fsync when the reads start"
        );

        let started = Instant::now();
        let mut slowest = Duration::ZERO;
        let mut reads = 0u32;
        while started.elapsed() < Duration::from_millis(500) {
            let t = Instant::now();
            assert!(store.get_project(other.id).await.unwrap().is_some());
            // The update is NOT visible before its fsync: the previous name is served.
            assert_eq!(
                store.get_project(p.id).await.unwrap().unwrap().name,
                p.name,
                "an update is published only once durable"
            );
            assert!(store
                .get_project_by_slug(&other.slug)
                .await
                .unwrap()
                .is_some());
            store.list_projects().await.unwrap();
            slowest = slowest.max(t.elapsed());
            reads += 1;
        }
        assert!(
            !update.is_finished() && !create.is_finished(),
            "the writes must still be waiting for their fsync while we read"
        );
        assert!(
            reads >= 1,
            "no read ran while the writes were in their fsync"
        );
        assert!(
            slowest < Duration::from_millis(1_000),
            "a read took {slowest:?} while writes waited 2 s for their fsync"
        );
        // A creation is visible only once durable.
        assert!(
            store.get_project(created.id).await.unwrap().is_none(),
            "a project is not visible before its fsync"
        );
        update.await.unwrap().unwrap();
        create.await.unwrap().unwrap();
        assert!(store.get_project(created.id).await.unwrap().is_some());
        assert_eq!(
            store.get_project(p.id).await.unwrap().unwrap().name,
            "renamed under fsync"
        );
    }

    // ------------------------------------------------------------------
    // Close
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn a_closed_store_refuses_every_operation_and_the_base_opens_again() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("closed");
        store.create_project(&p).await.unwrap();
        store.close().await.unwrap();
        store.close().await.unwrap(); // idempotent

        let refused = [
            store.get_project(p.id).await.map(|_| ()),
            store.get_project_by_slug(&p.slug).await.map(|_| ()),
            store.path_policy(p.id).await.map(|_| ()),
            store.list_projects().await.map(|_| ()),
            store.create_project(&a_project("after close")).await,
            store
                .update_project(p.id, Some("x".into()), None, None, None)
                .await,
            store.update_project_synced(p.id).await,
            store.update_project_analytics_timestamp(p.id).await,
            store.delete_project(p.id, "closed").await,
        ];
        assert_eq!(refused.len(), 9, "every method of ProjectStore");
        for r in refused {
            let msg = format!("{:#}", r.unwrap_err());
            assert!(msg.contains("fermée"), "{msg}");
        }
        drop(store);
        let reopened = open(&base).await;
        assert!(reopened.get_project(p.id).await.unwrap().is_some());
    }

    // ------------------------------------------------------------------
    // Single writer
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn a_base_already_open_for_writing_cannot_be_opened_again() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let first = open(&base).await;

        let second = try_open(&base, Creation::Interdite).await;
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
    // Other PROCESSES: the exclusive lock, and kill -9
    //
    // The test re-executes its own binary as the child (`--exact`, with the
    // directory in the environment). The same function is parent and child, so
    // it is one test that always runs — never a child case that passes when
    // run alone.
    // ------------------------------------------------------------------

    const ENV_VERROU: &str = "OBRAIN_PO_TEST_ENFANT_VERROU";
    const ENV_KILL9: &str = "OBRAIN_PO_TEST_ENFANT_KILL9";

    fn lancer_enfant(
        cas: &str,
        env: &str,
        dir: &Path,
        autres: &[(&str, &str)],
    ) -> std::process::Child {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args([cas, "--exact", "--nocapture", "--test-threads=1"])
            .env(env, dir)
            .envs(autres.iter().copied())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn the child")
    }

    async fn attendre(marqueur: &Path, quoi: &str) {
        for _ in 0..1200 {
            if marqueur.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("the child never wrote {quoi}");
    }

    /// The exclusive lock is obrain's, and it is what makes PO the single writer
    /// ACROSS processes. A second process is refused; when the first one dies
    /// (kill -9, no close) the lock is released by the system.
    #[tokio::test]
    async fn another_process_cannot_open_the_base_and_a_dead_one_releases_it() {
        if let Ok(dir) = std::env::var(ENV_VERROU) {
            let dir = std::path::PathBuf::from(dir);
            let store = open(&dir.join("base")).await;
            let p = a_project("held by the child");
            store.create_project(&p).await.unwrap();
            // Written whole, then renamed: the parent reads the marker as soon as it exists.
            std::fs::write(dir.join("pret.tmp"), p.id.to_string()).unwrap();
            std::fs::rename(dir.join("pret.tmp"), dir.join("pret")).unwrap();
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let mut child = lancer_enfant(
            "with_the_feature::another_process_cannot_open_the_base_and_a_dead_one_releases_it",
            ENV_VERROU,
            dir.path(),
            &[],
        );
        attendre(&dir.path().join("pret"), "its marker").await;
        let held_id: Uuid = std::fs::read_to_string(dir.path().join("pret"))
            .unwrap()
            .parse()
            .unwrap();

        let refused = try_open(&dir.path().join("base"), Creation::Interdite).await;
        let msg = match refused {
            Ok(_) => {
                child.kill().ok();
                panic!("a second PROCESS opened a base held for writing")
            }
            Err(e) => format!("{e:#}"),
        };
        assert!(msg.contains("ouverture de la base obrain"), "{msg}");

        child.kill().unwrap(); // SIGKILL: no close, no flush
        child.wait().unwrap();
        let store = open(&dir.path().join("base")).await;
        assert!(
            store.get_project(held_id).await.unwrap().is_some(),
            "what the dead holder acknowledged is there"
        );
    }

    /// A process killed with SIGKILL in the middle of creating projects, over and
    /// over. Two kinds of rounds, alternating:
    ///
    /// - `wal` (production durability): every project the child ACKNOWLEDGED is
    ///   there, whole, after the kill — long descriptions (blob column) and short
    ///   ones alike;
    /// - `aucune` (no fsync, no flush: the child is mostly INSIDE a creation when
    ///   the kill lands): nothing acknowledged is promised, but the base must
    ///   reopen, repair what was interrupted, quarantine nothing, and list.
    ///
    /// This is the process-crash model (the page cache survives). What a power cut
    /// does is obrain's guarantee.
    #[tokio::test]
    async fn kill9_during_creations_loses_no_acknowledged_project_and_never_blocks_the_reopen() {
        if let Ok(dir) = std::env::var(ENV_KILL9) {
            let dir = std::path::PathBuf::from(dir);
            let mode = std::env::var("OBRAIN_PO_TEST_ENFANT_MODE").unwrap();
            let durabilite = if mode == "wal" {
                Durabilite::Wal
            } else {
                Durabilite::Aucune
            };
            let store = {
                let base = dir.join("base");
                tokio::task::spawn_blocking(move || {
                    ObrainGraphStore::open_avec(base, Creation::Permise, durabilite)
                })
                .await
                .unwrap()
                .unwrap()
            };
            let mut log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(format!("acquittes-{mode}")))
                .unwrap();
            let mut i = 0u32;
            loop {
                let mut p = a_project("kill9");
                p.name = format!("kill9 {}", p.id);
                p.description = Some(description_du_test(p.id, i, mode == "wal"));
                i += 1;
                store.create_project(&p).await.unwrap();
                use std::io::Write as _;
                writeln!(log, "{}\t{}", p.id, i - 1).unwrap();
            }
        }

        const TOURS: usize = 12;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (mut repares, mut quarantaine) = (RapportOuverture::default(), 0usize);
        let (mut vus_wal, mut longues) = (0usize, 0usize);
        for tour in 0..TOURS {
            let mode = if tour % 2 == 0 { "wal" } else { "aucune" };
            let acquittes = dir.path().join(format!("acquittes-{mode}"));
            let compte = || std::fs::read_to_string(&acquittes).map_or(0, |s| s.lines().count());
            let avant = compte();
            let mut child = lancer_enfant(
                "with_the_feature::kill9_during_creations_loses_no_acknowledged_project_and_never_blocks_the_reopen",
                ENV_KILL9,
                dir.path(),
                &[("OBRAIN_PO_TEST_ENFANT_MODE", mode)],
            );
            // Wait until the child is creating (loudly: a child that never gets there
            // is a failure, not a round that silently kills nothing), then a delay that
            // varies per round so the kill lands at different points of a creation.
            for _ in 0..2400 {
                if compte() >= avant + 5 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            assert!(
                compte() >= avant + 5,
                "round {tour} ({mode}): the child acknowledged no creation in 60 s"
            );
            tokio::time::sleep(Duration::from_micros(1_500 * (tour as u64 % 9 + 1))).await;
            child.kill().unwrap();
            child.wait().unwrap();

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            repares.brouillons_supprimes += rapport.brouillons_supprimes;
            repares.chemins_repares += rapport.chemins_repares;
            quarantaine += rapport.en_quarantaine.len();
            assert!(
                rapport.en_quarantaine.is_empty(),
                "round {tour} ({mode}): a creation killed half-way must be REPAIRED, never \
                 quarantined: {rapport:?}"
            );

            if mode == "wal" {
                let log = std::fs::read_to_string(&acquittes).unwrap();
                for ligne in log.lines() {
                    let (id, i) = ligne.split_once('\t').unwrap();
                    let (id, i): (Uuid, u32) = (id.parse().unwrap(), i.parse().unwrap());
                    let p = store.get_project(id).await.unwrap().unwrap_or_else(|| {
                        panic!("round {tour}: acknowledged project {id} is gone")
                    });
                    assert_eq!(p.name, format!("kill9 {id}"));
                    assert_eq!(p.slug, format!("obrain-{id}"));
                    let attendue = description_du_test(id, i, true);
                    longues += usize::from(attendue.len() > 256);
                    assert_eq!(
                        p.description.as_deref(),
                        Some(attendue.as_str()),
                        "round {tour}: description of {id} ({} bytes) must survive the kill",
                        attendue.len()
                    );
                    vus_wal += 1;
                }
            }
            // Whatever the round was, every project the base lists is WHOLE: a creation
            // killed after its ninth property would still decode (the last two fields have
            // defaults), with `watch_enabled` true where the child wrote false.
            for p in store.list_projects().await.unwrap() {
                let id = p.id;
                assert_eq!(p.name, format!("kill9 {id}"), "round {tour} ({mode})");
                assert_eq!(p.slug, format!("obrain-{id}"), "round {tour} ({mode})");
                assert!(
                    !p.watch_enabled && p.profile == ProjectProfile::Software && p.root_path.is_empty(),
                    "round {tour} ({mode}): project {id} is torn (defaults where the child wrote values): {p:?}"
                );
                // Without durability (`aucune`) a blob-sized value may legitimately be gone;
                // what is there must still be the child's.
                match (mode, p.description.as_deref()) {
                    (_, Some(d)) => assert!(
                        d.contains(&id.to_string()),
                        "round {tour} ({mode}): project {id} carries another description"
                    ),
                    ("aucune", None) => {}
                    (_, None) => panic!("round {tour} ({mode}): project {id} lost its description"),
                }
            }
            store.close().await.unwrap();
        }
        assert!(vus_wal >= 30, "the child made progress ({vus_wal} checks)");
        assert!(longues > 0, "long descriptions were part of the check");
        // At least one kill landed inside a creation and its draft was found and
        // deleted: without this, a run where every kill fell between two creations
        // would prove nothing about the marker. (The exact step of each kill is
        // `an_interrupted_write_at_every_step_reopens_to_a_consistent_base`.)
        assert!(
            repares.brouillons_supprimes > 0,
            "no kill landed inside a creation over {TOURS} rounds: the marker was never exercised"
        );
        eprintln!(
            "kill -9 over {TOURS} rounds: {vus_wal} acknowledged-project checks ({longues} with a \
             description past the blob threshold); at open: {} interrupted creations deleted, {} \
             torn paths repaired, {quarantaine} quarantined",
            repares.brouillons_supprimes, repares.chemins_repares
        );
    }

    const ENV_KILL9_AUTH: &str = "OBRAIN_PO_TEST_ENFANT_KILL9_AUTH";

    /// The auth domain under SIGKILL, never closed: the child registers users and
    /// gives each a refresh token and an MCP token, logging each user once all three
    /// writes are acknowledged; it is killed while writing. The base reopens (from
    /// what the kill left, with no close and no full flush), quarantines nothing,
    /// deletes what was half-created, and serves everything acknowledged: the user
    /// by its key, the refresh token valid, the MCP token active.
    #[tokio::test]
    async fn kill9_during_auth_writes_loses_nothing_acknowledged() {
        if let Ok(dir) = std::env::var(ENV_KILL9_AUTH) {
            let dir = std::path::PathBuf::from(dir);
            let store = open(&dir.join("base")).await;
            let mut log = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("acquittes-auth"))
                .unwrap();
            let futur = Utc::now() + chrono::Duration::days(1);
            loop {
                let k = Uuid::new_v4();
                let u = store
                    .create_password_user(&format!("{k}@kill9.test"), "K", &format!("h-{k}"))
                    .await
                    .unwrap();
                store
                    .create_refresh_token(u.id, &format!("rt-{k}"), futur)
                    .await
                    .unwrap();
                store
                    .create_mcp_token(u.id, &format!("jti-{k}"), "l", "s", futur)
                    .await
                    .unwrap();
                use std::io::Write as _;
                writeln!(log, "{k}\t{}", u.id).unwrap();
            }
        }

        const TOURS: usize = 4;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let acquittes = dir.path().join("acquittes-auth");
        let compte = || std::fs::read_to_string(&acquittes).map_or(0, |s| s.lines().count());
        let mut vus = 0usize;
        for tour in 0..TOURS {
            let avant = compte();
            let mut child = lancer_enfant(
                "with_the_feature::kill9_during_auth_writes_loses_nothing_acknowledged",
                ENV_KILL9_AUTH,
                dir.path(),
                &[],
            );
            for _ in 0..2400 {
                if compte() >= avant + 5 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            assert!(
                compte() >= avant + 5,
                "round {tour}: the child acknowledged nothing in 60 s"
            );
            tokio::time::sleep(Duration::from_micros(1_300 * (tour as u64 % 7 + 1))).await;
            child.kill().unwrap(); // SIGKILL: no close, no flush
            child.wait().unwrap();

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert!(
                rapport.en_quarantaine.is_empty(),
                "round {tour}: an auth write killed half-way is repaired, never quarantined: \
                 {rapport:?}"
            );
            for ligne in std::fs::read_to_string(&acquittes).unwrap().lines() {
                let (k, id) = ligne.split_once('\t').unwrap();
                let id: Uuid = id.parse().unwrap();
                let u = store
                    .get_user_by_email_and_provider(&format!("{k}@kill9.test"), "password")
                    .await
                    .unwrap()
                    .unwrap_or_else(|| panic!("round {tour}: acknowledged user {id} is gone"));
                assert_eq!(u.id, id);
                assert_eq!(u.password_hash, Some(format!("h-{k}")));
                assert!(
                    store
                        .validate_refresh_token(&format!("rt-{k}"))
                        .await
                        .unwrap()
                        .is_some(),
                    "round {tour}: acknowledged refresh token of {id} is gone"
                );
                assert!(
                    store
                        .is_mcp_token_active(&format!("jti-{k}"))
                        .await
                        .unwrap(),
                    "round {tour}: acknowledged MCP token of {id} is gone"
                );
                vus += 1;
            }
            store.close().await.unwrap();
        }
        assert!(vus >= 20, "the child made progress ({vus} checks)");
    }

    /// Short, then long (multi-byte, past the 256-byte blob threshold), in turn. The
    /// `aucune` rounds only ever write SHORT values: a blob-sized write that is never
    /// flushed leaves an index row in the page cache pointing past the data the engine
    /// recovers, and a LATER write reuses that space — the first project then reads
    /// another project's description (measured, round 2 of this very test; an obrain
    /// defect, see the PR). `aucune` rounds exist to be killed inside a creation, not
    /// to exercise that.
    fn description_du_test(id: Uuid, i: u32, avec_blob: bool) -> String {
        if !avec_blob || i.is_multiple_of(3) {
            format!("short {id}")
        } else {
            format!("{id} é-long ").repeat(12 + (i as usize % 5))
        }
    }

    // ------------------------------------------------------------------
    // A base this backend did not write, or left half-written
    // ------------------------------------------------------------------

    /// The eleven properties a creation writes, with a root that exists on no
    /// machine (its canonical form is its declared form).
    fn complete(id: Uuid, name: &str, slug: &str, root: &str) -> Vec<(&'static str, Value)> {
        vec![
            ("id", Value::from(id.to_string())),
            ("name", Value::from(name)),
            ("slug", Value::from(slug)),
            ("root_path", Value::from(root)),
            ("canonical_path", Value::from(root)),
            ("declared_path", Value::from(root)),
            ("paths_declared", Value::from(true)),
            ("description", Value::from("d")),
            ("created_at", Value::from("2026-01-02T03:04:05+00:00")),
            ("watch_enabled", Value::from(false)),
            ("profile", Value::from("software")),
        ]
    }

    /// Every way a base can come to this backend not clean, in one base: opening
    /// repairs what is ours, quarantines what is not, and never fails.
    #[tokio::test]
    async fn opening_repairs_interrupted_writes_and_quarantines_what_it_cannot_address() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (a, d, e, g) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        let (a2, d2, e2, g2) = (
            format!("a-{a}"),
            format!("d-{d}"),
            format!("e-{e}"),
            format!("g-{g}"),
        );
        let (a2c, d2c, e2c, g2c) = (a2.clone(), d2.clone(), e2.clone(), g2.clone());
        write_foreign(&base, move |db| {
            // A: complete, but killed before the marker was removed: no commit point,
            //    not a project (deleted, like B and C).
            let mut props = complete(a, "complete with marker", &a2c, "/nonexistent/a");
            props.push(("_brouillon", Value::from(true)));
            db.create_node_with_props(&["Project"], props);
            // B: a creation killed after two properties (draft).
            db.create_node_with_props(
                &["Project"],
                vec![
                    ("_brouillon", Value::from(true)),
                    ("id", Value::from(Uuid::new_v4().to_string())),
                    ("name", Value::from("half written")),
                ],
            );
            // C: a creation killed right after the node was created.
            db.create_node(&["Project"]);
            // D: an update killed after the derived paths, before `root_path`:
            //    the declared root says old, root_path says new... or the reverse.
            let mut torn = complete(d, "torn update", &d2c, "/nonexistent/new");
            for (k, v) in torn.iter_mut() {
                if *k == "declared_path" || *k == "canonical_path" {
                    *v = Value::from("/nonexistent/old");
                }
            }
            db.create_node_with_props(&["Project"], torn);
            // E: two nodes claiming one id (foreign data).
            db.create_node_with_props(&["Project"], complete(e, "first", &e2c, "/nonexistent/e"));
            db.create_node_with_props(
                &["Project"],
                complete(
                    e,
                    "second, same id",
                    &format!("{e2c}-bis"),
                    "/nonexistent/e",
                ),
            );
            // F: foreign, no id: not ours, not a draft.
            db.create_node_with_props(&["Project"], vec![("name", Value::from("no id at all"))]);
            // G: a root with no canonical form at all (another writer's doing: no
            //    interruption of ours leaves this), caught by the second pure check.
            let mut sans_canonique = complete(g, "no canonical", &g2c, "/nonexistent/g");
            sans_canonique.retain(|(k, _)| *k != "canonical_path");
            db.create_node_with_props(&["Project"], sans_canonique);
        })
        .await;

        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert_eq!(rapport.brouillons_supprimes, 3, "A, B and C: {rapport:?}");
        assert_eq!(rapport.chemins_repares, 2, "D and G: {rapport:?}");
        assert_eq!(
            rapport.en_quarantaine.len(),
            2,
            "E's duplicate and F: {rapport:?}"
        );

        // What was repaired reads whole; the torn update rolled to a state
        // `root_path` states.
        let listed = store.list_projects().await.unwrap();
        assert_eq!(listed.len(), 3, "D, the first of E, and G");
        assert!(
            store.get_project(a).await.unwrap().is_none()
                && store.get_project_by_slug(&a2).await.unwrap().is_none(),
            "A never reached its commit point: not a project, its slug is free"
        );
        let pg = store.path_policy(g).await.unwrap();
        assert_eq!(pg.declared_root, "/nonexistent/g");
        assert_eq!(
            pg.canonical_root, "/nonexistent/g",
            "G's canonical form recomputed"
        );
        let _ = g2;
        let pd = store.get_project(d).await.unwrap().unwrap();
        assert_eq!(pd.root_path, "/nonexistent/new");
        let policy = store.path_policy(d).await.unwrap();
        assert_eq!(policy.declared_root, "/nonexistent/new");
        assert_eq!(policy.canonical_root, "/nonexistent/new");
        assert_eq!(store.get_project(e).await.unwrap().unwrap().name, "first");
        assert_eq!(
            store.get_project_by_slug(&d2).await.unwrap().map(|p| p.id),
            Some(d)
        );
        let _ = e2;
        // The store works on top of that.
        let p = a_project("after the repairs");
        store.create_project(&p).await.unwrap();
        store.close().await.unwrap();
        drop(store);

        // The repairs were durable: a second open has nothing left to repair,
        // and the two foreign nodes are still quarantined, untouched.
        let store = open(&base).await;
        let again = store.rapport_d_ouverture().clone();
        assert_eq!(again.brouillons_supprimes, 0, "{again:?}");
        assert_eq!(
            again.chemins_repares, 0,
            "a repaired project is not rewritten at every open: {again:?}"
        );
        assert_eq!(again.en_quarantaine.len(), 2, "{again:?}");
        assert_eq!(store.list_projects().await.unwrap().len(), 4);
    }

    /// A `Project` node written straight through obrain, with a relationship to
    /// a node of a domain this backend has not ported.
    #[tokio::test]
    async fn a_project_written_outside_is_read_and_its_cascade_delete_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let id = Uuid::new_v4();
        let slug = format!("migrated-{id}");
        {
            let slug = slug.clone();
            write_foreign(&base, move |db| {
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
                // Linked by a property, not an edge: what the cascade also reaches.
                db.create_node_with_props(
                    &["Commit"],
                    vec![("project_id", Value::from(id.to_string()))],
                );
            })
            .await;
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

        // The base holds nodes this backend did not write — by edge (Plan) AND
        // by property (Commit): refused by name, and nothing is deleted.
        let err = store.delete_project(id, "migrated").await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<NonPorte>(),
            Some(&NonPorte("ProjectStore::delete_project (cascade)")),
            "{err:#}"
        );
        assert!(
            format!("{err:#}").contains("2 nœud(s) hors des 1 projet"),
            "{err:#}"
        );
        assert!(store.get_project(id).await.unwrap().is_some());
    }

    // ------------------------------------------------------------------
    // The write ORDER, step by step: every point where a write can stop
    // ------------------------------------------------------------------

    /// What a base reopens to when a write stopped at a precise point. This is the
    /// deterministic half of the kill -9 test: the hook stops the write where the
    /// kill could have, the base is closed (what the engine wrote so far reaches the
    /// disk, exactly as the page cache would survive the kill), and opening is
    /// judged on that exact state.
    ///
    /// - create, stopped anywhere BEFORE the marker is removed: the draft is deleted
    ///   at open, the project does not exist, its slug is free, the writer saw an
    ///   error and nothing was published (all-or-nothing);
    /// - create, stopped AFTER the marker is removed but before the fsync: the
    ///   project exists at open although the writer got an error — the declared
    ///   "committed, acknowledgement lost" window; a retry is refused as a duplicate;
    /// - update of the root, stopped after `declared_path`, or after `canonical_path`:
    ///   opening rolls both derived paths back to what `root_path` (still old) states;
    ///   stopped after `root_path`: the new root, consistent, nothing to repair;
    /// - delete, stopped before the fsync: still served before the restart (not
    ///   published), gone after it (the same window as the creation's).
    #[tokio::test]
    async fn an_interrupted_write_at_every_step_reopens_to_a_consistent_base() {
        let dir = tempfile::tempdir().unwrap();
        let points_de_creation: Vec<Interruption> =
            [Interruption::ApresLeNoeud, Interruption::ApresLeMarqueur]
                .into_iter()
                .chain((1..=11).map(Interruption::ApresLaPropriete))
                .chain([
                    Interruption::AvantLeRetraitDuMarqueur,
                    Interruption::ApresLeRetraitDuMarqueur,
                ])
                .collect();
        for (n, point) in points_de_creation.iter().copied().enumerate() {
            let base = dir.path().join(format!("create-{n}"));
            let store = open(&base).await;
            let witness = a_project("witness");
            store.create_project(&witness).await.unwrap();

            let mut p = a_project("interrupted");
            // Past the blob threshold: the full flush before the marker is exercised too.
            p.description = Some("é-long ".repeat(60));
            store.interrompre_au_prochain(point);
            let err = store.create_project(&p).await.unwrap_err();
            assert!(
                format!("{err:#}").contains("interruption injectée"),
                "{point:?}: {err:#}"
            );
            // Nothing published, the store still serves (an interruption is a crash
            // of the write, not of the store).
            assert!(
                store.get_project(p.id).await.unwrap().is_none(),
                "{point:?}"
            );
            assert!(store.get_project(witness.id).await.unwrap().is_some());
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert!(rapport.en_quarantaine.is_empty(), "{point:?}: {rapport:?}");
            assert_eq!(rapport.chemins_repares, 0, "{point:?}: {rapport:?}");
            let listed = store.list_projects().await.unwrap();
            if point == Interruption::ApresLeRetraitDuMarqueur {
                // The commit point was passed: the project exists, whole, although the
                // writer never got its Ok.
                assert_eq!(rapport.brouillons_supprimes, 0, "{point:?}: {rapport:?}");
                assert_eq!(
                    fields(&store.get_project(p.id).await.unwrap().expect("promoted")),
                    fields(&p),
                    "{point:?}: the project is whole"
                );
                assert_eq!(listed.len(), 2);
                let again = store.create_project(&p).await.unwrap_err();
                assert!(format!("{again:#}").contains("existe déjà"), "{again:#}");
            } else {
                assert_eq!(
                    rapport.brouillons_supprimes, 1,
                    "{point:?}: the draft is found and deleted: {rapport:?}"
                );
                assert!(
                    store.get_project(p.id).await.unwrap().is_none(),
                    "{point:?}"
                );
                assert!(
                    store.get_project_by_slug(&p.slug).await.unwrap().is_none(),
                    "{point:?}: the slug is free"
                );
                assert_eq!(listed.len(), 1, "{point:?}: only the witness");
                // The retry succeeds: nothing was kept.
                store.create_project(&p).await.unwrap();
                assert_eq!(
                    fields(&store.get_project(p.id).await.unwrap().unwrap()),
                    fields(&p)
                );
            }
            store.close().await.unwrap();
        }

        // Update of the root, after each of the three writes.
        for n in 1..=3 {
            let base = dir.path().join(format!("update-{n}"));
            let store = open(&base).await;
            let mut p = a_project("rooted");
            p.root_path = "/nonexistent/old".into();
            store.create_project(&p).await.unwrap();

            store.interrompre_au_prochain(Interruption::ApresLeChemin(n));
            let err = store
                .update_project(p.id, None, None, Some("/nonexistent/new".into()), None)
                .await
                .unwrap_err();
            assert!(
                format!("{err:#}").contains("interruption injectée"),
                "{err:#}"
            );
            // Not published: the old root is served until the write is durable.
            assert_eq!(
                store.get_project(p.id).await.unwrap().unwrap().root_path,
                "/nonexistent/old"
            );
            let avant_fermeture = store.proprietes_brutes(p.id);
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            let apres_reouverture = store.proprietes_brutes(p.id);
            eprintln!(
                "after write {n}: before close {avant_fermeture:?}; after reopen {apres_reouverture:?}; {rapport:?}"
            );
            let read = store.get_project(p.id).await.unwrap().unwrap();
            let policy = store.path_policy(p.id).await.unwrap();
            let expected_root = if n == 3 {
                "/nonexistent/new"
            } else {
                "/nonexistent/old"
            };
            assert_eq!(
                rapport.chemins_repares,
                usize::from(n < 3),
                "after write {n}: a torn trio is repaired, a complete one is not: {rapport:?}; \
                 before close {avant_fermeture:?}; after reopen {apres_reouverture:?}"
            );
            assert_eq!(read.root_path, expected_root, "after write {n}");
            assert_eq!(
                policy.declared_root, expected_root,
                "after write {n}: declared"
            );
            assert_eq!(
                policy.canonical_root, expected_root,
                "after write {n}: canonical"
            );
            store.close().await.unwrap();
        }

        // Delete, before the fsync.
        let base = dir.path().join("delete");
        let store = open(&base).await;
        let p = a_project("deleted");
        store.create_project(&p).await.unwrap();
        store.interrompre_au_prochain(Interruption::ApresLaSuppression);
        let err = store.delete_project(p.id, "x").await.unwrap_err();
        assert!(
            format!("{err:#}").contains("interruption injectée"),
            "{err:#}"
        );
        assert!(
            store.get_project(p.id).await.unwrap().is_some(),
            "a deletion is published only once durable"
        );
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        assert!(
            store.get_project(p.id).await.unwrap().is_none(),
            "the deletion record was written: gone after the restart"
        );
    }

    // ------------------------------------------------------------------
    // The rest of the contract refuses by name
    // ------------------------------------------------------------------

    /// One method per kind of signature and per corner of the contract: first
    /// and last trait of `GraphStore`, a method the contract defaults to a
    /// silent `Ok`, the search store, and the default of `GraphStore` itself.
    #[tokio::test]
    async fn what_is_not_ported_refuses_by_name_and_never_answers_empty() {
        let store = temporary(Durabilite::Wal).await;
        let id = Uuid::new_v4();

        assert_eq!(
            not_ported(store.get_workspace_milestone_progress(id).await),
            NonPorte("WorkspaceStore::get_workspace_milestone_progress")
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
            not_ported(store.propagate_structural_links(id).await),
            NonPorte("NoteStore::propagate_structural_links")
        );
        // Defaulted by the contract to `Ok((0, 0))`: refused all the same.
        assert_eq!(
            not_ported(store.ensure_project_canonical_paths().await),
            NonPorte("AnalyticsStore::ensure_project_canonical_paths")
        );
        // The default of `GraphStore` answers `Ok({})` for an empty `ids` — the
        // silent empty result this backend must never give. It is written by
        // hand and refuses, with an empty list as with any other.
        assert_eq!(
            not_ported(store.get_progress_batch(ProgressKind::Project, &[]).await),
            NonPorte("GraphStore::get_progress_batch")
        );
        assert_eq!(
            not_ported(store.get_progress_batch(ProgressKind::Project, &[id]).await),
            NonPorte("GraphStore::get_progress_batch")
        );
        assert_eq!(
            not_ported(ObrainSearchStore.delete_code("x").await),
            NonPorte("SearchStore::delete_code")
        );

        // The message is the stated one.
        let msg = store
            .get_workspace_milestone_progress(Uuid::new_v4())
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "non porté sur le backend obrain : WorkspaceStore::get_workspace_milestone_progress"
        );
    }

    /// A protocol is started, a run walks it from state to state, a runtime state is
    /// generated: all of it — the protocol, its states and transition, the run at its
    /// current state with its history and version, the runtime state — survives a
    /// close and a reopen. Then the cascade of `delete_protocol` is durable too.
    #[tokio::test]
    async fn a_protocol_run_advances_and_is_found_in_its_state_after_a_reopen() {
        use project_orchestrator::protocol::{
            Protocol, ProtocolRun, ProtocolState, ProtocolTransition, RunStatus, RuntimeState,
            StateType, StateVisit,
        };
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;

        let project = a_project("protocol owner");
        store.create_project(&project).await.unwrap();
        let start = ProtocolState::start(Uuid::new_v4(), "start");
        let work = ProtocolState::new(start.protocol_id, "work");
        let mut done = ProtocolState::new(start.protocol_id, "done");
        done.state_type = StateType::Terminal;
        done.state_timeout_secs = Some(90);
        let mut protocol = Protocol::new(Some(project.id), "e2e protocol", start.id);
        protocol.id = start.protocol_id;
        protocol.terminal_states = vec![done.id];
        store.upsert_protocol(&protocol).await.unwrap();
        for s in [&start, &work, &done] {
            store.upsert_protocol_state(s).await.unwrap();
        }
        let go = ProtocolTransition::new(protocol.id, start.id, work.id, "begin");
        let finish = ProtocolTransition::new(protocol.id, work.id, done.id, "finish");
        for t in [&go, &finish] {
            store.upsert_protocol_transition(t).await.unwrap();
        }

        // Start the run, then advance it twice (each update is a compare-and-swap).
        let mut run = ProtocolRun::new(protocol.id, start.id, "start");
        run.project_id = Some(project.id);
        store.create_protocol_run(&run).await.unwrap();
        // A second root run is refused while the first one runs, and nothing is written.
        let second = ProtocolRun::new(protocol.id, start.id, "start");
        let refus = store.create_protocol_run(&second).await.unwrap_err();
        assert!(
            refus.to_string().contains("Skipped: concurrent run"),
            "{refus:#}"
        );
        assert!(store.get_protocol_run(second.id).await.unwrap().is_none());
        let visit = |state: &ProtocolState, trigger: &str| StateVisit {
            state_id: state.id,
            state_name: state.name.clone(),
            entered_at: Utc::now(),
            exited_at: None,
            duration_ms: None,
            trigger: Some(trigger.to_string()),
            progress_snapshot: None,
        };
        run.current_state = work.id;
        run.states_visited.push(visit(&work, "begin"));
        store.update_protocol_run(&mut run).await.unwrap();
        let runtime = RuntimeState::new(run.id, work.id, "generated", 0);
        store.create_runtime_state(&runtime).await.unwrap();
        run.current_state = done.id;
        run.states_visited.push(visit(&done, "finish"));
        run.status = RunStatus::Completed;
        run.completed_at = Some(Utc::now());
        store.update_protocol_run(&mut run).await.unwrap();
        assert_eq!(run.version, 2);
        // None runs now.
        assert!(store
            .find_active_run_for_project(project.id)
            .await
            .unwrap()
            .is_none());
        store.close().await.unwrap();

        let store = open(&base).await;
        assert_eq!(
            store.rapport_d_ouverture(),
            &RapportOuverture::default(),
            "nothing to repair, nothing quarantined"
        );
        let found = store.get_protocol_run(run.id).await.unwrap().unwrap();
        assert_eq!(
            found.current_state, done.id,
            "the run is in the state it reached"
        );
        assert_eq!(found.status, RunStatus::Completed);
        assert_eq!(found.version, 2);
        assert_eq!(found.project_id, Some(project.id));
        let path: Vec<&str> = found
            .states_visited
            .iter()
            .map(|v| v.state_name.as_str())
            .collect();
        assert_eq!(path, ["start", "work", "done"]);
        assert_eq!(found.states_visited[2].trigger.as_deref(), Some("finish"));
        assert_eq!(
            store
                .get_protocol(protocol.id)
                .await
                .unwrap()
                .unwrap()
                .terminal_states,
            vec![done.id]
        );
        let states = store.get_protocol_states(protocol.id).await.unwrap();
        assert_eq!(states.len(), 3);
        assert_eq!(
            states
                .iter()
                .find(|s| s.id == done.id)
                .unwrap()
                .state_timeout_secs,
            Some(90)
        );
        assert_eq!(
            states
                .iter()
                .find(|s| s.id == start.id)
                .unwrap()
                .state_timeout_secs,
            None
        );
        let triggers: Vec<String> = store
            .get_protocol_transitions(protocol.id)
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.trigger)
            .collect();
        assert_eq!(triggers, ["begin", "finish"]);
        let runtime_states = store.get_runtime_states(run.id).await.unwrap();
        assert_eq!(runtime_states.len(), 1);
        assert_eq!(runtime_states[0].generated_by, work.id);
        assert_eq!(
            store
                .list_completed_runs_for_project(project.id, 10)
                .await
                .unwrap()
                .len(),
            1
        );
        // The version survived: a stale writer still loses after the reopen.
        let mut stale = found.clone();
        stale.version = 1;
        let lost = store.update_protocol_run(&mut stale).await.unwrap_err();
        assert!(
            lost.to_string().contains("OptimisticLockError"),
            "the stale writer loses its compare-and-swap, not anything else: {lost:#}"
        );

        // The cascade is durable: protocol, states, transitions, runs, runtime states.
        assert!(store.delete_protocol(protocol.id).await.unwrap());
        store.close().await.unwrap();
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        assert!(store.get_protocol(protocol.id).await.unwrap().is_none());
        assert!(store.get_protocol_run(run.id).await.unwrap().is_none());
        assert!(store.get_runtime_states(run.id).await.unwrap().is_empty());
        assert!(store
            .get_protocol_states(protocol.id)
            .await
            .unwrap()
            .is_empty());
        // Nothing of the protocol domain is left in the base: the project can go.
        store
            .delete_project(project.id, &project.name)
            .await
            .unwrap();
        store.close().await.unwrap();
    }

    /// What of `ProtocolRunStore` is not ported refuses by name; `ProtocolStore`
    /// refuses nothing.
    #[tokio::test]
    async fn the_run_methods_that_name_other_domains_refuse_by_name() {
        let store = temporary(Durabilite::Wal).await;
        let id = Uuid::new_v4();
        let tree = project_orchestrator::reasoning::ReasoningTree::new("why", None);
        // What each refused method ACTUALLY answers, by calling it.
        let answered = [
            not_ported(store.get_run_outcomes(id).await),
            not_ported(store.create_produced_during("note", id, id).await),
            not_ported(store.get_run_reasoning_tree_id(id).await),
            not_ported(
                store
                    .persist_reasoning_tree(&tree, Some("protocol_run"), Some(id))
                    .await,
            ),
        ];
        assert_eq!(
            answered,
            [
                NonPorte("ProtocolRunStore::get_run_outcomes"),
                NonPorte("ProtocolRunStore::create_produced_during"),
                NonPorte("ProtocolRunStore::get_run_reasoning_tree_id"),
                NonPorte("ProtocolRunStore::persist_reasoning_tree"),
            ]
        );
        // The registry the accounting reads names exactly the refusals the calls
        // gave: one more, or one fewer, is caught here.
        let mut registry: Vec<&str> = refusees_protocolrun().to_vec();
        let mut called: Vec<&str> = answered.iter().map(|n| n.0).collect();
        registry.sort_unstable();
        called.sort_unstable();
        assert_eq!(registry, called);
        assert!(!methodes_refusees()
            .iter()
            .any(|m| m.starts_with("ProtocolStore::")));
    }

    /// The commit methods that are not ported refuse by name, with their reason.
    #[tokio::test]
    async fn commit_refusals_are_named() {
        let store = temporary(Durabilite::Wal).await;
        let id = Uuid::new_v4();
        assert_eq!(
            not_ported(store.find_dependent_files("a.rs", 3, None).await),
            NonPorte("CommitStore::find_dependent_files")
        );
        assert_eq!(
            not_ported(store.find_impacted_files("a.rs", 3, None).await),
            NonPorte("CommitStore::find_impacted_files")
        );
        assert_eq!(
            not_ported(store.find_callers("a.rs::f", None).await),
            NonPorte("CommitStore::find_callers")
        );
        assert_eq!(
            not_ported(store.link_task_to_files(id, &[]).await),
            NonPorte("CommitStore::link_task_to_files")
        );
        assert_eq!(
            not_ported(store.link_commit_to_task("h", id).await),
            NonPorte("CommitStore::link_commit_to_task")
        );
        assert_eq!(
            not_ported(store.link_commit_to_task_from("h", id, "merged").await),
            NonPorte("CommitStore::link_commit_to_task_from")
        );
        assert_eq!(
            not_ported(store.get_task_commits(id).await),
            NonPorte("CommitStore::get_task_commits")
        );
        assert_eq!(
            not_ported(store.ping_freshness_for_files(&[]).await),
            NonPorte("CommitStore::ping_freshness_for_files")
        );
        // The TOUCHES / CO_CHANGED machinery: no production code writes `File`
        // on obrain, so served it would answer Ok while dropping every file.
        let p = Uuid::new_v4();
        assert_eq!(
            not_ported(
                store
                    .create_commit_touches("h", &[touche("a.rs", Some(1), Some(1))])
                    .await
            ),
            NonPorte("CommitStore::create_commit_touches")
        );
        assert_eq!(
            not_ported(store.get_commit_files("h").await),
            NonPorte("CommitStore::get_commit_files")
        );
        assert_eq!(
            not_ported(store.get_file_history("a.rs", None).await),
            NonPorte("CommitStore::get_file_history")
        );
        assert_eq!(
            not_ported(store.compute_co_changed(p, None, 1, 100).await),
            NonPorte("CommitStore::compute_co_changed")
        );
        assert_eq!(
            not_ported(store.get_co_change_graph(p, 1, 100).await),
            NonPorte("CommitStore::get_co_change_graph")
        );
        assert_eq!(
            not_ported(store.get_file_co_changers("a.rs", 1, 100).await),
            NonPorte("CommitStore::get_file_co_changers")
        );
        assert_eq!(
            not_ported(store.compute_co_changed_transitive(p, 2, 0.1).await),
            NonPorte("CommitStore::compute_co_changed_transitive")
        );
        assert_eq!(
            not_ported(
                store
                    .get_file_transitive_co_changers("a.rs", 0.1, 100)
                    .await
            ),
            NonPorte("CommitStore::get_file_transitive_co_changers")
        );
    }

    fn commit(hash: &str, secs: i64) -> CommitNode {
        CommitNode {
            hash: hash.to_string(),
            message: format!("message {hash}"),
            author: "e2e".to_string(),
            timestamp: chrono::DateTime::from_timestamp(1_700_000_000 + secs, 0).unwrap(),
        }
    }

    fn touche(path: &str, additions: Option<i64>, deletions: Option<i64>) -> FileChangedInfo {
        FileChangedInfo {
            path: path.to_string(),
            additions,
            deletions,
        }
    }

    /// The commit domain end to end, as it serves: commits, their plan link and
    /// the project's co-change timestamp — written, then read back AFTER a close
    /// and a reopen. The TOUCHES / CO_CHANGED machinery is refused (no `File` is
    /// written in production); its own close-and-reopen proof is a module test
    /// (`commit.rs`, `co_change_transitif_et_horodatage_survivent_a_la_reouverture`).
    #[tokio::test]
    async fn commits_and_their_plan_link_survive_a_close_and_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;

        let projet = a_project("commit");
        store.create_project(&projet).await.unwrap();
        let plan = PlanNode {
            id: Uuid::new_v4(),
            title: "plan commit".to_string(),
            description: "description".to_string(),
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            created_by: "e2e".to_string(),
            priority: 1,
            project_id: Some(projet.id),
            execution_context: None,
            persona: None,
        };
        store.create_plan(&plan).await.unwrap();

        let c1 = commit("h1", 1);
        let c2 = commit("h2", 2);
        store.create_commit(&c1).await.unwrap();
        store.create_commit(&c2).await.unwrap();
        store.link_commit_to_plan("h1", plan.id).await.unwrap();
        store.link_commit_to_plan("h2", plan.id).await.unwrap();
        // A second link: the plan's list stays one entry per commit. That the
        // ENGINE holds one edge (the MERGE) is counted in the module test
        // `commit.rs::commit_lie_a_un_plan`: the index is a set and would hide it.
        store.link_commit_to_plan("h1", plan.id).await.unwrap();
        store
            .update_project_co_change_timestamp(projet.id)
            .await
            .unwrap();
        let horodatage = store
            .get_project(projet.id)
            .await
            .unwrap()
            .unwrap()
            .last_co_change_computed_at
            .expect("le projet est horodaté");

        store.close().await.unwrap();
        let store = open(&base).await;

        assert_eq!(
            store.get_commit("h1").await.unwrap().unwrap().message,
            "message h1"
        );
        let commits = store.get_plan_commits(plan.id).await.unwrap();
        assert_eq!(
            commits.iter().map(|c| c.hash.as_str()).collect::<Vec<_>>(),
            ["h2", "h1"],
            "newest first, each once"
        );
        assert_eq!(
            store
                .get_project(projet.id)
                .await
                .unwrap()
                .unwrap()
                .last_co_change_computed_at,
            Some(horodatage)
        );

        store.delete_commit("h1").await.unwrap();
        assert!(store.get_commit("h1").await.unwrap().is_none());
        let commits = store.get_plan_commits(plan.id).await.unwrap();
        assert_eq!(
            commits.iter().map(|c| c.hash.as_str()).collect::<Vec<_>>(),
            ["h2"],
            "the deleted commit leaves the plan's list"
        );
    }

    /// A note with a project, a link to the project and to another note, a
    /// succession and a confirmation: all of it survives a close and a reopen, with
    /// the same values, the same links and the same order.
    #[tokio::test]
    async fn notes_survive_a_close_and_a_reopen_with_links_and_states() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{EntityType, Note, NoteFilters, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("notes-reopen");
        store.create_project(&p).await.unwrap();
        let mut n = Note::new(
            Some(p.id),
            NoteType::Gotcha,
            format!(
                "une note longue, au-delà du seuil de colonne : {}",
                "é".repeat(200)
            ),
            "alice".to_string(),
        );
        n.tags = vec!["a".into(), "b".into()];
        let m = Note::new(
            None,
            NoteType::Tip,
            "globale".to_string(),
            "system/x".to_string(),
        );
        store.create_note(&n).await.unwrap();
        store.create_note(&m).await.unwrap();
        store
            .link_note_to_entity(
                n.id,
                &EntityType::Project,
                &p.id.to_string(),
                Some("sig"),
                Some("body"),
            )
            .await
            .unwrap();
        store
            .link_note_to_entity(n.id, &EntityType::Note, &m.id.to_string(), None, None)
            .await
            .unwrap();
        store.supersede_note(m.id, n.id).await.unwrap();
        let confirmee = store.confirm_note(n.id, "alice").await.unwrap().unwrap();
        let avant_n = store.get_note(n.id).await.unwrap().unwrap();
        let avant_m = store.get_note(m.id).await.unwrap().unwrap();
        let avant_liste = store
            .list_notes(Some(p.id), None, &NoteFilters::default())
            .await
            .unwrap();
        let avant_anchors = store.get_note_anchors(n.id).await.unwrap();
        assert_eq!(avant_n.human_confirmed_by.as_deref(), Some("alice"));
        assert_eq!(avant_n.human_confirmed_by, confirmee.human_confirmed_by);
        assert_eq!(
            avant_m.status,
            project_orchestrator::notes::NoteStatus::Archived
        );
        assert_eq!(avant_anchors.len(), 2);
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        assert_eq!(
            store.get_note(n.id).await.unwrap().unwrap().content,
            avant_n.content
        );
        assert_eq!(
            store.get_note(n.id).await.unwrap().unwrap().tags,
            avant_n.tags
        );
        assert_eq!(
            store.get_note(m.id).await.unwrap().unwrap().superseded_by,
            Some(n.id)
        );
        let liste = store
            .list_notes(Some(p.id), None, &NoteFilters::default())
            .await
            .unwrap();
        assert_eq!(liste.0.len(), avant_liste.0.len());
        assert_eq!(
            liste.0.iter().map(|x| x.id).collect::<Vec<_>>(),
            avant_liste.0.iter().map(|x| x.id).collect::<Vec<_>>()
        );
        assert_eq!(store.get_note_anchors(n.id).await.unwrap().len(), 2);
        assert_eq!(
            store
                .get_notes_for_entity(&EntityType::Note, &m.id.to_string())
                .await
                .unwrap()
                .iter()
                .map(|x| x.id)
                .collect::<Vec<_>>(),
            vec![n.id]
        );
    }

    /// A creation of a note interrupted at every step reopens to a consistent base:
    /// the draft is deleted with its edges, or the note is whole when the commit
    /// point was passed. Nothing is quarantined.
    #[tokio::test]
    async fn an_interrupted_note_creation_reopens_consistently() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let mut points = vec![Interruption::ApresLeNoeud, Interruption::ApresLeMarqueur];
        points.extend((1..=40).map(Interruption::ApresLaPropriete));
        points.extend([
            Interruption::AvantLeRetraitDuMarqueur,
            Interruption::ApresLeRetraitDuMarqueur,
        ]);
        let mut touches = 0;
        for (k, point) in points.iter().copied().enumerate() {
            let base = dir.path().join(format!("note-{k}"));
            let store = open(&base).await;
            let p = a_project("note-interrupted");
            store.create_project(&p).await.unwrap();
            let n = Note::new(
                Some(p.id),
                NoteType::Tip,
                "interrompue".to_string(),
                "alice".to_string(),
            );
            store.interrompre_au_prochain(point);
            let issue = store.create_note(&n).await;
            if issue.is_ok() {
                // The creation finished before this point: the point is not on its path.
                store.close().await.unwrap();
                continue;
            }
            touches += 1;
            assert!(store.get_note(n.id).await.unwrap().is_none(), "{point:?}");
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert!(rapport.en_quarantaine.is_empty(), "{point:?}: {rapport:?}");
            if point == Interruption::ApresLeRetraitDuMarqueur {
                assert!(store.get_note(n.id).await.unwrap().is_some(), "{point:?}");
            } else {
                assert_eq!(rapport.brouillons_supprimes, 1, "{point:?}: {rapport:?}");
                assert!(store.get_note(n.id).await.unwrap().is_none(), "{point:?}");
                assert!(store
                    .list_notes(Some(p.id), None, &Default::default())
                    .await
                    .unwrap()
                    .0
                    .is_empty());
                store.create_note(&n).await.unwrap();
            }
            store.close().await.unwrap();
        }
        // Every creation step is on the path: the properties, the marker, the commit point.
        assert!(touches >= 28, "the points were reached: {touches}");
    }

    /// What the note domain refuses, refuses by name and before any write: an
    /// agent author, a workspace filter, a link to an entity of another domain.
    #[tokio::test]
    async fn note_refusals_are_named_and_write_nothing() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{EntityType, Note, NoteFilters, NoteType};
        let store = temporary(Durabilite::Wal).await;
        let p = a_project("note-refus");
        store.create_project(&p).await.unwrap();

        let agent = Note::new(
            Some(p.id),
            NoteType::Tip,
            "écrite par un agent".to_string(),
            format!("agent:{}", Uuid::new_v4()),
        );
        let err = store.create_note(&agent).await.unwrap_err();
        assert_eq!(
            err.downcast_ref::<NonPorte>(),
            Some(&NonPorte("NoteStore::create_note"))
        );
        assert!(store.get_note(agent.id).await.unwrap().is_none());

        let f = NoteFilters::default();
        assert_eq!(
            not_ported(store.list_notes(None, Some("ws"), &f).await),
            NonPorte("NoteStore::list_notes")
        );
        assert_eq!(
            not_ported(store.get_notes_needing_review(None, Some("ws")).await),
            NonPorte("NoteStore::get_notes_needing_review")
        );
        let n = Note::new(
            Some(p.id),
            NoteType::Tip,
            "ok".to_string(),
            "alice".to_string(),
        );
        store.create_note(&n).await.unwrap();
        assert_eq!(
            not_ported(
                store
                    .link_note_to_entity(n.id, &EntityType::File, "src/lib.rs", None, None)
                    .await
            ),
            NonPorte("NoteStore::link_note_to_entity")
        );
        assert!(store.get_note_anchors(n.id).await.unwrap().is_empty());
        // A link to a project this backend does not hold is a no-op, as in Neo4j.
        store
            .link_note_to_entity(
                n.id,
                &EntityType::Project,
                &Uuid::new_v4().to_string(),
                None,
                None,
            )
            .await
            .unwrap();
        assert!(store.get_note_anchors(n.id).await.unwrap().is_empty());
    }

    /// A multi-write is a sequence of engine writes with no marker. A crash just before
    /// the k-th write (`AvantEtape(k)`: the k-1 first ones are on disk) must reopen with nothing quarantined, and a retry of the same call must
    /// converge to the full state. One loop per operation (supersede, rescope, delete,
    /// update, confirm, link).
    #[tokio::test]
    async fn note_multi_writes_interrupted_then_retried_converge() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{
            EntityType, Note, NoteFilters, NoteImportance, NoteStatus, NoteType,
        };
        let dir = tempfile::tempdir().unwrap();

        // A base with one project and two notes, crashed at the k-th write of `nom`.
        async fn fixture(dir: &Path, nom: &str, k: usize) -> (ObrainGraphStore, Uuid, Note, Note) {
            let store = open(&dir.join(format!("{nom}-{k}"))).await;
            let p = a_project("multi");
            store.create_project(&p).await.unwrap();
            let a = Note::new(Some(p.id), NoteType::Tip, "a".into(), "alice".into());
            let b = Note::new(Some(p.id), NoteType::Tip, "b".into(), "alice".into());
            store.create_note(&a).await.unwrap();
            store.create_note(&b).await.unwrap();
            (store, p.id, a, b)
        }
        // Reopen after the crash, check that nothing is quarantined.
        async fn reouvrir(dir: &Path, nom: &str, k: usize) -> ObrainGraphStore {
            let store = open(&dir.join(format!("{nom}-{k}"))).await;
            assert!(
                store.rapport_d_ouverture().en_quarantaine.is_empty(),
                "{nom} k={k}"
            );
            store
        }
        let mut atteints = 0;

        // supersede: old archived + superseded_by, new.supersedes, the edge.
        for k in 1..=10 {
            let (store, _p, a, b) = fixture(dir.path(), "supersede", k).await;
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.supersede_note(a.id, b.id).await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            assert!(format!("{:#}", issue.unwrap_err()).contains("interruption injectée"));
            store.close().await.unwrap();
            drop(store);
            let store = reouvrir(dir.path(), "supersede", k).await;
            store.supersede_note(a.id, b.id).await.unwrap();
            let (va, vb) = (
                store.get_note(a.id).await.unwrap().unwrap(),
                store.get_note(b.id).await.unwrap().unwrap(),
            );
            assert_eq!(
                (va.status, va.superseded_by),
                (NoteStatus::Archived, Some(b.id)),
                "k={k}"
            );
            assert_eq!(vb.supersedes, Some(a.id), "k={k}");
            store.close().await.unwrap();
        }

        // rescope to the global level: the HAS_NOTE edge goes, the property goes.
        for k in 1..=10 {
            let (store, p, a, _b) = fixture(dir.path(), "rescope", k).await;
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.rescope_note(a.id, None).await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            store.close().await.unwrap();
            drop(store);
            let store = reouvrir(dir.path(), "rescope", k).await;
            store.rescope_note(a.id, None).await.unwrap();
            assert_eq!(
                store.get_note(a.id).await.unwrap().unwrap().project_id,
                None,
                "k={k}"
            );
            let (dans_p, _) = store
                .list_notes(Some(p), None, &NoteFilters::default())
                .await
                .unwrap();
            assert!(!dans_p.iter().any(|n| n.id == a.id), "k={k}");
            store.close().await.unwrap();
        }

        // delete: edges first, then the node. A crash between them leaves a note with no
        // edges; the retry deletes it.
        for k in 1..=10 {
            let (store, _p, a, _b) = fixture(dir.path(), "delete", k).await;
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.delete_note(a.id).await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            store.close().await.unwrap();
            drop(store);
            let store = reouvrir(dir.path(), "delete", k).await;
            store.delete_note(a.id).await.unwrap();
            assert!(store.get_note(a.id).await.unwrap().is_none(), "k={k}");
            store.close().await.unwrap();
        }

        // update: the fields are written one by one.
        for k in 1..=10 {
            let (store, _p, a, _b) = fixture(dir.path(), "update", k).await;
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store
                .update_note(
                    a.id,
                    Some("nouveau".into()),
                    Some(NoteImportance::Critical),
                    None,
                    None,
                    None,
                )
                .await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            store.close().await.unwrap();
            drop(store);
            let store = reouvrir(dir.path(), "update", k).await;
            store
                .update_note(
                    a.id,
                    Some("nouveau".into()),
                    Some(NoteImportance::Critical),
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();
            let v = store.get_note(a.id).await.unwrap().unwrap();
            assert_eq!(
                (v.content.as_str(), v.importance),
                ("nouveau", NoteImportance::Critical),
                "k={k}"
            );
            store.close().await.unwrap();
        }

        // confirm: the freshness clock and the signature, field by field.
        for k in 1..=12 {
            let (store, _p, a, _b) = fixture(dir.path(), "confirm", k).await;
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.confirm_note(a.id, "alice").await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            store.close().await.unwrap();
            drop(store);
            let store = reouvrir(dir.path(), "confirm", k).await;
            let v = store.confirm_note(a.id, "alice").await.unwrap().unwrap();
            assert_eq!(v.human_confirmed_by.as_deref(), Some("alice"), "k={k}");
            assert_eq!(v.staleness_score, 0.0, "k={k}");
            store.close().await.unwrap();
        }

        // link: the edge is created, then its three properties. A crash after the edge
        // leaves a link with no hashes; the retry gives it its hashes (no duplicate edge).
        for k in 1..=6 {
            let (store, _p, a, b) = fixture(dir.path(), "link", k).await;
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store
                .link_note_to_entity(
                    a.id,
                    &EntityType::Note,
                    &b.id.to_string(),
                    Some("sig"),
                    Some("body"),
                )
                .await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            store.close().await.unwrap();
            drop(store);
            let store = reouvrir(dir.path(), "link", k).await;
            store
                .link_note_to_entity(
                    a.id,
                    &EntityType::Note,
                    &b.id.to_string(),
                    Some("sig"),
                    Some("body"),
                )
                .await
                .unwrap();
            let anchors = store.get_note_anchors(a.id).await.unwrap();
            assert_eq!(anchors.len(), 1, "one link after the retry, k={k}");
            assert_eq!(anchors[0].signature_hash.as_deref(), Some("sig"), "k={k}");
            store.close().await.unwrap();
        }
        // supersede 4, rescope 2, delete 2, update 3, confirm 8, link 3 writes.
        assert_eq!(atteints, 22, "the crash points were reached");
    }

    /// An id whose node is on disk but unreadable at open is reserved: a new note may not
    /// take it (two nodes would share an id).
    #[tokio::test]
    async fn an_id_on_disk_but_unreadable_is_reserved_for_notes() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        store.close().await.unwrap();
        drop(store);
        let id = Uuid::new_v4();
        // A Note node with an id and no `created_at`: unreadable at open, so quarantined.
        write_foreign(&base, move |db| {
            db.create_node_with_props(
                &["Note"],
                vec![
                    ("id", Value::from(id.to_string())),
                    ("status", Value::from("active")),
                    ("note_type", Value::from("tip")),
                    ("content", Value::from("illisible")),
                    ("created_by", Value::from("alice")),
                ],
            );
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        let mut n = Note::new(None, NoteType::Tip, "nouvelle".into(), "alice".into());
        n.id = id;
        let err = store.create_note(&n).await.unwrap_err();
        assert!(format!("{err:#}").contains("réservé"), "{err:#}");
        assert!(store.get_note(id).await.unwrap().is_none());
    }

    /// A base with links to a domain this backend does not port: the anchors of that note
    /// are a named refusal, not a silently short list.
    #[tokio::test]
    async fn anchors_are_refused_by_name_when_the_base_links_to_an_unported_domain() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let n = Note::new(
            None,
            NoteType::Tip,
            "liée à un fichier".into(),
            "alice".into(),
        );
        store.create_note(&n).await.unwrap();
        let autre = Note::new(None, NoteType::Tip, "sans lien".into(), "alice".into());
        store.create_note(&autre).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let id = n.id;
        write_foreign(&base, move |db| {
            let noeud = db.find_nodes_by_property("id", &Value::from(id.to_string()))[0];
            let fichier = db.create_node(&["File"]);
            db.set_node_property(fichier, "path", Value::from("/src/lib.rs"));
            db.create_edge(noeud, fichier, "LINKED_TO");
        })
        .await;
        let store = open(&base).await;
        assert_eq!(
            not_ported(store.get_note_anchors(id).await),
            NonPorte("NoteStore::get_note_anchors")
        );
        assert!(store.get_note_anchors(autre.id).await.unwrap().is_empty());
    }

    /// Entity ids of a ported type must be canonical: two spellings of one node would be
    /// two links. Refused by name, before any write.
    #[tokio::test]
    async fn non_canonical_entity_ids_are_refused_before_any_write() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{EntityType, Note, NoteType};
        let store = temporary(Durabilite::Wal).await;
        let p = a_project("canonique");
        store.create_project(&p).await.unwrap();
        let n = Note::new(Some(p.id), NoteType::Tip, "x".into(), "alice".into());
        store.create_note(&n).await.unwrap();
        let majuscules = p.id.to_string().to_uppercase();
        for err in [
            store
                .link_note_to_entity(n.id, &EntityType::Project, &majuscules, None, None)
                .await
                .unwrap_err(),
            store
                .unlink_note_from_entity(n.id, &EntityType::Project, &majuscules)
                .await
                .unwrap_err(),
            store
                .get_notes_for_entity(&EntityType::Project, &majuscules)
                .await
                .unwrap_err(),
        ] {
            assert!(format!("{err:#}").contains("non canonique"), "{err:#}");
        }
        assert!(store.get_note_anchors(n.id).await.unwrap().is_empty());
        // The canonical spelling twice: one link.
        for _ in 0..2 {
            store
                .link_note_to_entity(n.id, &EntityType::Project, &p.id.to_string(), None, None)
                .await
                .unwrap();
        }
        assert_eq!(store.get_note_anchors(n.id).await.unwrap().len(), 1);
    }

    /// A non-finite value of a sort key is refused at creation: NaN has no place in a total order.
    #[tokio::test]
    async fn a_non_finite_sort_key_is_refused_at_creation() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{Note, NoteType};
        let store = temporary(Durabilite::Wal).await;
        let mut n = Note::new(None, NoteType::Tip, "nan".into(), "alice".into());
        n.staleness_score = f64::NAN;
        let err = store.create_note(&n).await.unwrap_err();
        assert!(format!("{err:#}").contains("non fini"), "{err:#}");
        assert!(store.get_note(n.id).await.unwrap().is_none());
    }

    /// Decision noyau : une décision globale rattachée à un projet, une autre sans
    /// projet, une mise à jour, une supersession, puis une fermeture et une
    /// réouverture — tout est durable, y compris une description au-delà du seuil
    /// de colonne. Le jalon « décision de bout en bout » sur obrain.
    #[tokio::test]
    async fn decisions_survive_a_close_and_a_reopen_with_project_and_supersession() {
        use project_orchestrator::neo4j::models::{DecisionNode, DecisionStatus};
        use project_orchestrator::neo4j::traits::{DecisionStore, ProjectStore};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("decision-reopen");
        store.create_project(&p).await.unwrap();

        let d1 = DecisionNode {
            id: Uuid::new_v4(),
            description: format!(
                "une description longue, au-delà du seuil de colonne : {}",
                "é".repeat(200)
            ),
            rationale: "parce que".to_string(),
            alternatives: vec!["autre".to_string()],
            chosen_option: Some("A".to_string()),
            decided_by: "alice".to_string(),
            decided_at: Utc::now(),
            status: DecisionStatus::Proposed,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        };
        let d2 = DecisionNode {
            id: Uuid::new_v4(),
            description: "choix B".to_string(),
            rationale: "parce que aussi".to_string(),
            alternatives: vec![],
            chosen_option: None,
            decided_by: "alice".to_string(),
            decided_at: Utc::now(),
            status: DecisionStatus::Proposed,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        };

        store.create_decision_global(&d1, Some(p.id)).await.unwrap();
        store.create_decision_global(&d2, None).await.unwrap();

        assert_eq!(
            store
                .get_decision(d1.id)
                .await
                .unwrap()
                .unwrap()
                .description,
            d1.description
        );
        assert_eq!(
            store.get_decision_project_id(d1.id).await.unwrap(),
            Some(p.id.to_string())
        );
        assert_eq!(store.get_decision_project_id(d2.id).await.unwrap(), None);
        let projets = store
            .get_decision_project_ids(&[d1.id, d2.id])
            .await
            .unwrap();
        assert_eq!(projets.len(), 1);
        assert_eq!(projets.get(&d1.id), Some(&p.id));

        // The long description is NOT touched by the update: it is read back after
        // the reopen as it was written at creation.
        store
            .update_decision(
                d1.id,
                None,
                Some("nouveau pourquoi".to_string()),
                Some("B".to_string()),
                Some(DecisionStatus::Accepted),
            )
            .await
            .unwrap();
        let maj = store.get_decision(d1.id).await.unwrap().unwrap();
        assert_eq!(maj.description, d1.description);
        assert_eq!(maj.rationale, "nouveau pourquoi");
        assert_eq!(maj.chosen_option.as_deref(), Some("B"));
        assert_eq!(maj.status, DecisionStatus::Accepted);

        // Twice: the MERGE draws one edge.
        store.supersede_decision(d2.id, d1.id).await.unwrap();
        store.supersede_decision(d2.id, d1.id).await.unwrap();
        assert_eq!(
            store.get_decision(d1.id).await.unwrap().unwrap().status,
            DecisionStatus::Superseded
        );

        // Une décision rattachée à un projet inconnu est refusée, rien n'est créé.
        let inconnu = DecisionNode {
            id: Uuid::new_v4(),
            description: "x".to_string(),
            rationale: "y".to_string(),
            alternatives: vec![],
            chosen_option: None,
            decided_by: "alice".to_string(),
            decided_at: Utc::now(),
            status: DecisionStatus::Proposed,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        };
        let err = store
            .create_decision_global(&inconnu, Some(Uuid::new_v4()))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("projet inconnu"), "{err:#}");
        assert!(store.get_decision(inconnu.id).await.unwrap().is_none());

        store.close().await.unwrap();
        drop(store);

        let aretes = supersedes_on_disk(&base).await;
        assert_eq!(aretes.len(), 1, "two supersessions, one edge: {aretes:?}");
        let quand = aretes[0].as_deref().expect("the edge carries `created_at`");
        assert!(quand.parse::<chrono::DateTime<Utc>>().is_ok(), "{quand}");

        let store = open(&base).await;
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        let relue = store.get_decision(d1.id).await.unwrap().unwrap();
        assert_eq!(relue.description, d1.description);
        assert_eq!(relue.rationale, "nouveau pourquoi");
        assert_eq!(relue.status, DecisionStatus::Superseded);
        assert_eq!(relue.chosen_option.as_deref(), Some("B"));
        assert_eq!(
            store.get_decision_project_id(d1.id).await.unwrap(),
            Some(p.id.to_string())
        );
        assert_eq!(
            store
                .get_decision(d2.id)
                .await
                .unwrap()
                .unwrap()
                .description,
            "choix B"
        );
        assert_eq!(store.get_decision_project_id(d2.id).await.unwrap(), None);

        // Suppression : le DETACH emporte la décision et ses arêtes.
        store.delete_decision(d2.id).await.unwrap();
        assert!(store.get_decision(d2.id).await.unwrap().is_none());
        assert_eq!(
            store
                .get_decision_project_ids(&[d1.id])
                .await
                .unwrap()
                .get(&d1.id),
            Some(&p.id)
        );
    }

    /// Les méthodes non portées de `DecisionStore` refusent par leur nom, et n'écrivent
    /// rien.
    #[tokio::test]
    async fn decision_refusals_are_named_and_write_nothing() {
        use project_orchestrator::neo4j::models::{DecisionNode, DecisionStatus};
        use project_orchestrator::neo4j::traits::DecisionStore;
        let store = temporary(Durabilite::Wal).await;
        // A project and a decision already there: the count below is not trivially 0.
        let p = a_project("decision-refus");
        store.create_project(&p).await.unwrap();
        store
            .create_decision_global(&a_decision("déjà là"), Some(p.id))
            .await
            .unwrap();
        let avant = store.noeuds_pour_mesure();
        let d = DecisionNode {
            id: Uuid::new_v4(),
            description: "x".to_string(),
            rationale: "y".to_string(),
            alternatives: vec![],
            chosen_option: None,
            decided_by: "alice".to_string(),
            decided_at: Utc::now(),
            status: DecisionStatus::Proposed,
            embedding: None,
            embedding_model: None,
            scar_intensity: 0.0,
            sharing_consent: Default::default(),
        };
        assert_eq!(
            not_ported(store.create_decision(Uuid::new_v4(), &d).await),
            NonPorte("DecisionStore::create_decision")
        );
        assert_eq!(
            not_ported(store.set_decision_embedding(d.id, &[1.0f32], "m").await),
            NonPorte("DecisionStore::set_decision_embedding")
        );
        assert_eq!(
            not_ported(store.get_decision_embedding(d.id).await),
            NonPorte("DecisionStore::get_decision_embedding")
        );
        assert_eq!(
            not_ported(store.search_decisions_by_vector(&[1.0f32], 5, None).await),
            NonPorte("DecisionStore::search_decisions_by_vector")
        );
        assert_eq!(
            not_ported(store.get_decisions_without_embedding().await),
            NonPorte("DecisionStore::get_decisions_without_embedding")
        );
        assert_eq!(
            not_ported(store.get_all_decisions_with_task_id().await),
            NonPorte("DecisionStore::get_all_decisions_with_task_id")
        );
        assert_eq!(
            not_ported(store.get_decisions_for_entity("File", "src/x.rs", 5).await),
            NonPorte("DecisionStore::get_decisions_for_entity")
        );
        assert_eq!(
            not_ported(
                store
                    .get_decisions_affecting("File", "src/x.rs", None)
                    .await
            ),
            NonPorte("DecisionStore::get_decisions_affecting")
        );
        assert_eq!(
            not_ported(
                store
                    .add_decision_affects(d.id, "File", "src/x.rs", None)
                    .await
            ),
            NonPorte("DecisionStore::add_decision_affects")
        );
        assert_eq!(
            not_ported(
                store
                    .remove_decision_affects(d.id, "File", "src/x.rs")
                    .await
            ),
            NonPorte("DecisionStore::remove_decision_affects")
        );
        assert_eq!(
            not_ported(store.list_decision_affects(d.id).await),
            NonPorte("DecisionStore::list_decision_affects")
        );
        assert_eq!(
            not_ported(store.get_decision_timeline(None, None, None).await),
            NonPorte("DecisionStore::get_decision_timeline")
        );
        assert_eq!(
            not_ported(
                store
                    .list_decisions_by_status(DecisionStatus::Proposed, None, None, 10, 0)
                    .await
            ),
            NonPorte("DecisionStore::list_decisions_by_status")
        );
        // Les refus n'ont rien écrit : ni la décision, ni aucun autre nœud.
        assert!(store.get_decision(d.id).await.unwrap().is_none());
        assert_eq!(store.noeuds_pour_mesure(), avant);
    }

    /// A decision creation interrupted at each of its points reopens to no decision
    /// (the draft is deleted at open), except past the commit point; the creation can
    /// then be made again.
    #[tokio::test]
    async fn an_interrupted_decision_creation_reopens_consistently() {
        use project_orchestrator::neo4j::traits::DecisionStore;
        let dir = tempfile::tempdir().unwrap();
        let mut points = vec![Interruption::ApresLeNoeud, Interruption::ApresLeMarqueur];
        points.extend((1..=9).map(Interruption::ApresLaPropriete));
        points.extend([
            Interruption::AvantLeRetraitDuMarqueur,
            Interruption::ApresLeRetraitDuMarqueur,
        ]);
        let mut touches = 0;
        for (k, point) in points.iter().copied().enumerate() {
            let base = dir.path().join(format!("decision-{k}"));
            let store = open(&base).await;
            let p = a_project("decision-interrupted");
            store.create_project(&p).await.unwrap();
            let d = a_decision("interrompue");
            store.interrompre_au_prochain(point);
            let issue = store.create_decision_global(&d, Some(p.id)).await;
            assert!(issue.is_err(), "{point:?} is on the path of a creation");
            touches += 1;
            assert!(
                store.get_decision(d.id).await.unwrap().is_none(),
                "{point:?}"
            );
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert!(rapport.en_quarantaine.is_empty(), "{point:?}: {rapport:?}");
            if point == Interruption::ApresLeRetraitDuMarqueur {
                assert!(
                    store.get_decision(d.id).await.unwrap().is_some(),
                    "{point:?}"
                );
                assert_eq!(
                    store.get_decision_project_id(d.id).await.unwrap(),
                    Some(p.id.to_string())
                );
            } else {
                assert_eq!(rapport.brouillons_supprimes, 1, "{point:?}: {rapport:?}");
                assert!(
                    store.get_decision(d.id).await.unwrap().is_none(),
                    "{point:?}"
                );
                // Only the project is left: the draft went with its edge.
                assert_eq!(store.noeuds_pour_mesure(), 1, "{point:?}");
                store.create_decision_global(&d, Some(p.id)).await.unwrap();
            }
            store.close().await.unwrap();
        }
        assert_eq!(touches, 13);
    }

    /// `delete_decision` crashed after the k-th engine write: the reopen finishes the
    /// deletion (the draft marker goes first), or — crashed before any write — finds
    /// the decision whole. Never a decision stripped of part of its edges.
    #[tokio::test]
    async fn an_interrupted_decision_deletion_is_finished_at_open() {
        use project_orchestrator::neo4j::traits::DecisionStore;
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=10 {
            let base = dir.path().join(format!("delete-{k}"));
            let store = open(&base).await;
            let p = a_project("decision-delete");
            store.create_project(&p).await.unwrap();
            let (a, b, c) = (a_decision("a"), a_decision("b"), a_decision("c"));
            store.create_decision_global(&a, Some(p.id)).await.unwrap();
            store.create_decision_global(&b, None).await.unwrap();
            store.create_decision_global(&c, None).await.unwrap();
            // `a` has three edges: HAS_DECISION in, SUPERSEDES in (from b), out (to c).
            store.supersede_decision(b.id, a.id).await.unwrap();
            store.supersede_decision(a.id, c.id).await.unwrap();

            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.delete_decision(a.id).await;
            if issue.is_ok() {
                store.close().await.unwrap();
                break;
            }
            atteints += 1;
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert!(rapport.en_quarantaine.is_empty(), "k={k}: {rapport:?}");
            if k == 1 {
                // Nothing written yet: the decision is whole.
                assert!(store.get_decision(a.id).await.unwrap().is_some());
                assert_eq!(
                    store.get_decision_project_id(a.id).await.unwrap(),
                    Some(p.id.to_string())
                );
            } else {
                assert_eq!(rapport.brouillons_supprimes, 1, "k={k}: {rapport:?}");
                assert!(store.get_decision(a.id).await.unwrap().is_none(), "k={k}");
            }
            // The others are untouched, and a retry converges.
            assert!(store.get_decision(b.id).await.unwrap().is_some());
            assert!(store.get_decision(c.id).await.unwrap().is_some());
            store.delete_decision(a.id).await.unwrap();
            assert!(store.get_decision(a.id).await.unwrap().is_none());
            assert_eq!(store.noeuds_pour_mesure(), 3, "k={k}");
            store.close().await.unwrap();
            drop(store);
            assert!(supersedes_on_disk(&base).await.is_empty(), "k={k}");
        }
        // The marker, the three edges, the node.
        assert_eq!(atteints, 5);
    }

    /// Two readable Decision nodes with the same id: the second is quarantined and the
    /// id is reserved — a write on it is refused, named, and a creation too.
    #[tokio::test]
    async fn a_duplicate_decision_id_is_reserved_and_its_writes_refused() {
        use project_orchestrator::neo4j::traits::DecisionStore;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        store.close().await.unwrap();
        drop(store);
        let d = a_decision("en double");
        let id = d.id;
        write_foreign(&base, move |db| {
            for description in ["premier", "second"] {
                db.create_node_with_props(
                    &["Decision"],
                    vec![
                        ("id", Value::from(id.to_string())),
                        ("description", Value::from(description)),
                        ("rationale", Value::from("r")),
                        ("status", Value::from("proposed")),
                    ],
                );
            }
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        assert!(store.get_decision(id).await.unwrap().is_some());
        let err = store.delete_decision(id).await.unwrap_err();
        assert!(format!("{err:#}").contains("refusé"), "{err:#}");
        assert!(store.get_decision(id).await.unwrap().is_some());
        let err = store.create_decision_global(&d, None).await.unwrap_err();
        assert!(format!("{err:#}").contains("existe déjà"), "{err:#}");
        let avant = store.noeuds_pour_mesure();
        assert_eq!(avant, 2, "both nodes are still on disk");
    }

    /// A decision quarantined at open (unreadable: no `description`): update, delete
    /// and supersede on its id refuse by name, never answer `Ok` without acting.
    #[tokio::test]
    async fn writes_on_a_quarantined_decision_are_refused_not_ignored() {
        use project_orchestrator::neo4j::models::DecisionStatus;
        use project_orchestrator::neo4j::traits::DecisionStore;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let saine = a_decision("saine");
        store.create_decision_global(&saine, None).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let id = Uuid::new_v4();
        write_foreign(&base, move |db| {
            db.create_node_with_props(
                &["Decision"],
                vec![
                    ("id", Value::from(id.to_string())),
                    ("rationale", Value::from("sans description")),
                ],
            );
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        let refuse = |r: anyhow::Result<()>| {
            let e = r.expect_err("a write on a quarantined decision must refuse");
            assert!(format!("{e:#}").contains("quarantaine"), "{e:#}");
        };
        refuse(
            store
                .update_decision(id, None, None, None, Some(DecisionStatus::Accepted))
                .await,
        );
        refuse(store.delete_decision(id).await);
        refuse(store.supersede_decision(saine.id, id).await);
        refuse(store.supersede_decision(id, saine.id).await);
        // Nothing was written: the sound decision is untouched, both nodes remain.
        let relue = store.get_decision(saine.id).await.unwrap().unwrap();
        assert_eq!(relue.status, DecisionStatus::Proposed);
        assert_eq!(store.noeuds_pour_mesure(), 2);
        // An id that is neither held nor reserved stays Neo4j's silent no-op.
        store.delete_decision(Uuid::new_v4()).await.unwrap();
    }

    /// An update that puts a `chosen_option` above the blob threshold, or replaces
    /// one that sat there, pays the FULL flush: counted in fsyncs, with no
    /// `close()` (the WAL alone does not recover such a value).
    #[tokio::test]
    async fn a_long_chosen_option_pays_the_full_flush_without_close() {
        use project_orchestrator::neo4j::traits::DecisionStore;
        let store = temporary(Durabilite::Wal).await;
        let fsyncs = || store.fsyncs_du_wal().unwrap();
        let d = a_decision("courte");
        store.create_decision_global(&d, None).await.unwrap();
        let mut cout = Vec::new();
        for option in ["a".to_string(), "x".repeat(300), "b".to_string()] {
            let t = fsyncs();
            store
                .update_decision(d.id, None, None, Some(option), None)
                .await
                .unwrap();
            cout.push(fsyncs() - t);
        }
        let [court, long, remplace] = cout[..] else {
            unreachable!()
        };
        assert!(court >= 1, "the WAL fsync ran");
        assert!(long > court, "long chosen_option: {long} against {court}");
        assert!(
            remplace > court,
            "long chosen_option replaced: {remplace} against {court}"
        );
    }

    /// `supersede_decision` crashed after its k-th engine write (status, edge,
    /// `created_at`): the retry converges to ONE edge that carries its `created_at`.
    #[tokio::test]
    async fn an_interrupted_supersession_converges_with_its_date() {
        use project_orchestrator::neo4j::models::DecisionStatus;
        use project_orchestrator::neo4j::traits::DecisionStore;
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for k in 1..=4 {
            let base = dir.path().join(format!("supersede-{k}"));
            let store = open(&base).await;
            let (a, b) = (a_decision("ancienne"), a_decision("nouvelle"));
            store.create_decision_global(&a, None).await.unwrap();
            store.create_decision_global(&b, None).await.unwrap();
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let issue = store.supersede_decision(b.id, a.id).await;
            store.close().await.unwrap();
            drop(store);
            if issue.is_ok() {
                break;
            }
            atteints += 1;
            let store = open(&base).await;
            assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
            store.supersede_decision(b.id, a.id).await.unwrap();
            assert_eq!(
                store.get_decision(a.id).await.unwrap().unwrap().status,
                DecisionStatus::Superseded
            );
            store.close().await.unwrap();
            drop(store);
            let aretes = supersedes_on_disk(&base).await;
            assert_eq!(aretes.len(), 1, "k={k}: {aretes:?}");
            assert!(
                aretes[0].is_some(),
                "k={k}: the edge carries its created_at"
            );
        }
        // The status, the edge, its date.
        assert_eq!(atteints, 3);
    }

    /// A `HAS_DECISION` edge from a Project node quarantined as a duplicate id: the
    /// decision reads as having NO project, not the held project of the same id.
    #[tokio::test]
    async fn a_decision_of_a_duplicate_project_node_reads_without_project() {
        use project_orchestrator::neo4j::traits::DecisionStore;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("le vrai");
        store.create_project(&p).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let d = a_decision("rattachée au doublon");
        let (pid, did) = (p.id, d.id);
        write_foreign(&base, move |db| {
            let doublon = db.create_node_with_props(
                &["Project"],
                vec![
                    ("id", Value::from(pid.to_string())),
                    ("name", Value::from("doublon")),
                    ("slug", Value::from(format!("doublon-{pid}"))),
                    ("created_at", Value::from("2026-01-02T03:04:05+00:00")),
                ],
            );
            let decision = db.create_node_with_props(
                &["Decision"],
                vec![
                    ("id", Value::from(did.to_string())),
                    ("description", Value::from("d")),
                    ("rationale", Value::from("r")),
                    ("status", Value::from("proposed")),
                ],
            );
            db.create_edge(doublon, decision, "HAS_DECISION");
        })
        .await;
        let store = open(&base).await;
        // Precondition: the held project is the original, the duplicate is quarantined.
        assert_eq!(
            store.get_project(p.id).await.unwrap().unwrap().name,
            "le vrai"
        );
        assert_eq!(
            store.rapport_d_ouverture().en_quarantaine.len(),
            1,
            "{:?}",
            store.rapport_d_ouverture()
        );
        assert!(store.get_decision(d.id).await.unwrap().is_some());
        assert_eq!(store.get_decision_project_id(d.id).await.unwrap(), None);
        assert!(store
            .get_decision_project_ids(&[d.id])
            .await
            .unwrap()
            .is_empty());
    }

    /// A NaN stored by a migration: it sorts after every number whatever its sign bit (as
    /// Neo4j's `Double.compare`; `f64::total_cmp` puts a negative NaN first), and both bounds
    /// of the staleness filters exclude it, as Neo4j's `NaN >= x` does.
    #[tokio::test]
    async fn a_migrated_nan_note_sorts_last_and_is_excluded_by_both_bounds() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::NoteFilters;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        store.close().await.unwrap();
        drop(store);
        let (a, b, c, d) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        // `d` holds a NaN with its sign bit set.
        let nan_negatif = f64::from_bits(f64::NAN.to_bits() | (1 << 63));
        assert!(nan_negatif.is_nan() && nan_negatif.is_sign_negative());
        write_foreign(&base, move |db| {
            for (id, staleness) in [(a, 0.1), (b, f64::NAN), (c, 0.5), (d, nan_negatif)] {
                db.create_node_with_props(
                    &["Note"],
                    vec![
                        ("id", Value::from(id.to_string())),
                        ("status", Value::from("active")),
                        ("note_type", Value::from("tip")),
                        ("importance", Value::from("medium")),
                        ("content", Value::from("migrée")),
                        ("created_by", Value::from("alice")),
                        ("created_at", Value::from(Utc::now().to_rfc3339())),
                        ("staleness_score", Value::from(staleness)),
                    ],
                );
            }
        })
        .await;
        let store = open(&base).await;
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        let ids = |v: Vec<project_orchestrator::notes::Note>| {
            v.into_iter().map(|n| n.id).collect::<Vec<_>>()
        };
        let base_f = NoteFilters {
            global_only: Some(true),
            sort_by: Some("staleness_score".into()),
            sort_order: Some("asc".into()),
            ..Default::default()
        };
        // The two NaNs tie, the id breaks the tie.
        let (n1, n2) = if b < d { (b, d) } else { (d, b) };
        let (liste, _) = store.list_notes(None, None, &base_f).await.unwrap();
        assert_eq!(
            ids(liste),
            vec![a, c, n1, n2],
            "the NaNs are last, ascending"
        );
        let desc = NoteFilters {
            sort_order: Some("desc".into()),
            ..base_f.clone()
        };
        let (liste, _) = store.list_notes(None, None, &desc).await.unwrap();
        assert_eq!(
            ids(liste),
            vec![n2, n1, c, a],
            "the NaNs are first, descending"
        );
        let min = NoteFilters {
            min_staleness: Some(-1.0),
            ..base_f.clone()
        };
        assert_eq!(
            ids(store.list_notes(None, None, &min).await.unwrap().0),
            vec![a, c],
            "min excludes the NaNs"
        );
        let max = NoteFilters {
            max_staleness: Some(1.0),
            ..base_f.clone()
        };
        assert_eq!(
            ids(store.list_notes(None, None, &max).await.unwrap().0),
            vec![a, c],
            "max excludes the NaNs"
        );
        let serrée = NoteFilters {
            max_staleness: Some(0.2),
            ..base_f
        };
        assert_eq!(
            ids(store.list_notes(None, None, &serrée).await.unwrap().0),
            vec![a],
            "a finite bound still filters"
        );
    }

    /// Migrated ids: `ABC…` is the node `abc…`. A canonical link reuses the migrated edge
    /// (no second link). A base holding two edges to the same target refuses, by name, the
    /// anchors and the link / unlink of that target (the index keeps one edge: the other
    /// would stay on disk); deleting the target removes both edges and lifts the refusal.
    #[tokio::test]
    async fn a_migrated_uppercase_note_link_is_the_same_link_as_its_canonical_spelling() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{EntityType, Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let a = Note::new(None, NoteType::Tip, "a".into(), "alice".into());
        let b = Note::new(None, NoteType::Tip, "b".into(), "alice".into());
        store.create_note(&a).await.unwrap();
        store.create_note(&b).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let (ia, ib) = (a.id, b.id);
        // The migrated base: b's id is written in uppercase, and a has one edge to it.
        write_foreign(&base, move |db| {
            let na = db.find_nodes_by_property("id", &Value::from(ia.to_string()))[0];
            let nb = db.find_nodes_by_property("id", &Value::from(ib.to_string()))[0];
            db.set_node_property(nb, "id", Value::from(ib.to_string().to_uppercase()));
            db.create_edge(na, nb, "LINKED_TO");
        })
        .await;
        let store = open(&base).await;
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        store
            .link_note_to_entity(a.id, &EntityType::Note, &b.id.to_string(), Some("s"), None)
            .await
            .unwrap();
        assert_eq!(
            store.get_note_anchors(a.id).await.unwrap().len(),
            1,
            "the migrated edge is reused"
        );
        store.close().await.unwrap();
        drop(store);

        // A second edge to the same node: refused by name.
        write_foreign(&base, move |db| {
            let na = db.find_nodes_by_property("id", &Value::from(ia.to_string()))[0];
            let nb =
                db.find_nodes_by_property("id", &Value::from(ib.to_string().to_uppercase()))[0];
            db.create_edge(na, nb, "LINKED_TO");
        })
        .await;
        let store = open(&base).await;
        assert_eq!(
            not_ported(store.get_note_anchors(a.id).await),
            NonPorte("NoteStore::get_note_anchors")
        );
        assert_eq!(
            not_ported(
                store
                    .link_note_to_entity(a.id, &EntityType::Note, &b.id.to_string(), None, None)
                    .await
            ),
            NonPorte("NoteStore::link_note_to_entity")
        );
        assert_eq!(
            not_ported(
                store
                    .unlink_note_from_entity(a.id, &EntityType::Note, &b.id.to_string())
                    .await
            ),
            NonPorte("NoteStore::unlink_note_from_entity")
        );
        // Neo4j would answer the note twice (no DISTINCT): refused, not answered once.
        assert_eq!(
            not_ported(
                store
                    .get_notes_for_entity(&EntityType::Note, &b.id.to_string())
                    .await
            ),
            NonPorte("NoteStore::get_notes_for_entity")
        );
        // Deleting the target takes both edges (DETACH): nothing is left to refuse.
        assert!(store.delete_note(b.id).await.unwrap());
        assert!(store.get_note_anchors(a.id).await.unwrap().is_empty());
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        assert!(store.get_note_anchors(a.id).await.unwrap().is_empty());
    }

    /// The branch of a refused note deletion, after its edges are gone: the store says so,
    /// and refuses every later operation until a restart.
    #[tokio::test]
    async fn a_refused_note_deletion_after_its_edges_makes_the_base_indeterminate() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{Note, NoteType};
        let store = temporary(Durabilite::Wal).await;
        let p = a_project("indeterminee");
        store.create_project(&p).await.unwrap();
        let n = Note::new(Some(p.id), NoteType::Tip, "x".into(), "alice".into());
        store.create_note(&n).await.unwrap();
        store.interrompre_au_prochain(Interruption::SuppressionEchouee);
        let err = store.delete_note(n.id).await.unwrap_err();
        assert!(format!("{err:#}").contains("indéterminée"), "{err:#}");
        let after = store.get_note(n.id).await.unwrap_err();
        assert!(format!("{after:#}").contains("indéterminé"), "{after:#}");
    }

    /// A node quarantined at open because its id is already taken carries the id of an
    /// indexed note: an edge to it is not a link of that note (deleting the indexed note
    /// would leave the edge, and the quarantined node would come back as that note).
    #[tokio::test]
    async fn an_edge_to_a_quarantined_note_node_is_not_a_link_of_the_indexed_note() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let a = Note::new(None, NoteType::Tip, "a".into(), "alice".into());
        let t = Note::new(None, NoteType::Tip, "t".into(), "alice".into());
        store.create_note(&a).await.unwrap();
        store.create_note(&t).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let (ia, it) = (a.id, t.id);
        // A second node with t's id (another spelling), and a's only edge goes to it.
        write_foreign(&base, move |db| {
            let na = db.find_nodes_by_property("id", &Value::from(ia.to_string()))[0];
            let fantome = db.create_node_with_props(
                &["Note"],
                vec![
                    ("id", Value::from(it.to_string().to_uppercase())),
                    ("status", Value::from("active")),
                    ("note_type", Value::from("tip")),
                    ("importance", Value::from("medium")),
                    ("content", Value::from("fantôme")),
                    ("created_by", Value::from("alice")),
                    ("created_at", Value::from(Utc::now().to_rfc3339())),
                ],
            );
            db.create_edge(na, fantome, "LINKED_TO");
        })
        .await;
        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture();
        assert!(
            rapport
                .en_quarantaine
                .iter()
                .any(|q| q.contains("déjà pris")),
            "{:?}",
            rapport.en_quarantaine
        );
        assert_eq!(
            store.get_note(t.id).await.unwrap().unwrap().content,
            "t",
            "the indexed node is t's own"
        );
        assert_eq!(
            not_ported(store.get_note_anchors(a.id).await),
            NonPorte("NoteStore::get_note_anchors"),
            "the edge to the quarantined node is unresolved, not a link to t"
        );
    }

    /// An edge deletion the engine refuses, in `delete_note` or `unlink_note_from_entity`:
    /// the index would drop an edge the base still holds, so the store says it is
    /// indeterminate and refuses every later operation until a restart.
    #[tokio::test]
    async fn a_refused_note_edge_deletion_makes_the_base_indeterminate() {
        use project_orchestrator::neo4j::traits::NoteStore;
        use project_orchestrator::notes::{EntityType, Note, NoteType};
        // delete_note: the note's HAS_NOTE edge.
        let store = temporary(Durabilite::Wal).await;
        let p = a_project("arete-refusee");
        store.create_project(&p).await.unwrap();
        let n = Note::new(Some(p.id), NoteType::Tip, "x".into(), "alice".into());
        store.create_note(&n).await.unwrap();
        store.interrompre_au_prochain(Interruption::SuppressionDAreteEchouee);
        let err = store.delete_note(n.id).await.unwrap_err();
        assert!(format!("{err:#}").contains("indéterminée"), "{err:#}");
        let after = store.get_note(n.id).await.unwrap_err();
        assert!(format!("{after:#}").contains("indéterminé"), "{after:#}");

        // unlink_note_from_entity: the LINKED_TO edge.
        let store = temporary(Durabilite::Wal).await;
        let a = Note::new(None, NoteType::Tip, "a".into(), "alice".into());
        let b = Note::new(None, NoteType::Tip, "b".into(), "alice".into());
        store.create_note(&a).await.unwrap();
        store.create_note(&b).await.unwrap();
        store
            .link_note_to_entity(a.id, &EntityType::Note, &b.id.to_string(), None, None)
            .await
            .unwrap();
        store.interrompre_au_prochain(Interruption::SuppressionDAreteEchouee);
        let err = store
            .unlink_note_from_entity(a.id, &EntityType::Note, &b.id.to_string())
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("indéterminée"), "{err:#}");
        let after = store.get_note_anchors(a.id).await.unwrap_err();
        assert!(format!("{after:#}").contains("indéterminé"), "{after:#}");
    }

    /// The generator skips nothing: every `async fn` of the two contracts is
    /// either refused, or written by hand (and the build has checked that it IS
    /// written), or one of the few that cannot refuse and keep the contract's
    /// default.
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
        let written = methodes_portees();
        let kept = defauts_conserves();
        // `GraphStore::capabilities` is the one written method that is not `async`.
        let written_async = written
            .iter()
            .filter(|m| **m != "GraphStore::capabilities")
            .count();
        assert_eq!(
            refused.len() + written_async + kept.len(),
            in_contracts,
            "refused {} + written {} + kept {} must be the {} async fn of the contracts",
            refused.len(),
            written_async,
            kept.len(),
            in_contracts
        );

        // ── LE REGISTRE EST DÉRIVÉ, PLUS SNAPSHOTÉ (13/11/2026) ──
        //
        // Ce test portait des VALEURS LITTÉRALES : le nombre de méthodes qui servent,
        // la liste des domaines, leurs préfixes. Chaque lot de portage devait donc
        // réécrire ce fichier — et comme tous les lots y touchent, chaque PR de
        // domaine entrait en conflit avec toutes les autres (mesuré : trois fusions
        // sur cinq ont nécessité une résolution manuelle de ce seul fichier).
        //
        // Ces valeurs sont maintenant DÉRIVÉES de `PORTEES` / `REFUSEES`, que le
        // build génère (see `build/refus_obrain.rs`) : un lot ne peut plus mentir
        // sur son compte, et il n'a plus à éditer ce fichier.
        let servent = methodes_qui_servent();
        assert!(
            servent
                .iter()
                .all(|m| !refused.contains(m) && !REFUS_ECRITS_A_LA_MAIN.contains(m)),
            "une méthode ne peut pas servir ET être refusée : {servent:?}"
        );
        // Tout domaine qu'on prétend avoir écrit doit avoir son module dans
        // `src/obrain/` : un lot qui oublie de créer son fichier est attrapé ici,
        // sans aucune liste à tenir à la main.
        let domaines: std::collections::BTreeSet<&str> = written
            .iter()
            .map(|m| m.split("::").next().unwrap())
            .collect();
        for d in &domaines {
            if *d == "GraphStore" {
                continue;
            }
            let fichier = root
                .join("src/obrain")
                .join(format!("{}.rs", d.trim_end_matches("Store").to_lowercase()));
            assert!(
                fichier.exists(),
                "domaine `{d}` compté comme porté sans module `{}`",
                fichier.display()
            );
        }
        // The written refusals are written, and are not counted as serving.
        assert!(refusees_note().iter().all(|m| written.contains(m)));
        assert!(refusees_release().iter().all(|m| written.contains(m)));
        assert!(refusees_commit().iter().all(|m| written.contains(m)));
        assert!(refusees_chat().iter().all(|m| written.contains(m)));
        assert!(refusees_decision().iter().all(|m| written.contains(m)));
        assert!(refusees_plan().iter().all(|m| written.contains(m)));
        assert!(refusees_protocolrun().iter().all(|m| written.contains(m)));

        assert!(refusees_planrun().iter().all(|m| written.contains(m)));

        assert!(refusees_skill().iter().all(|m| written.contains(m)));
        assert!(refusees_sharing().iter().all(|m| written.contains(m)));
        assert!(REFUS_ECRITS_A_LA_MAIN.iter().all(|m| written.contains(m)));
        // What keeps a default is known and justified where it is declared.
        assert_eq!(kept, ["AnalyticsStore::run_data_migrations"]);
        // No name twice, no ported method refused.
        let mut all: Vec<&str> = refused.iter().chain(written).chain(kept).copied().collect();
        let total = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), total, "a method is listed twice");
        eprintln!(
            "obrain backend: {} methods refused by name, {} written by hand ({} serve), {} keeping the contract's default",
            refused.len(),
            written.len(),
            servent.len(),
            kept.len()
        );
    }

    /// Chat noyau : sessions, événements, ancres et réglages — écrits, relus,
    /// puis relus APRÈS une fermeture et une réouverture (persistance), et enfin
    /// supprimés. Le jalon « chat de bout en bout » sur obrain.
    #[tokio::test]
    async fn chat_noyau_survit_une_reouverture() {
        use project_orchestrator::chat::anchor::{
            AnchorActor, AnchorError, AnchorOp, AnchorRole, AnchorTargetType, NewAnchor,
        };
        use project_orchestrator::chat::provider::policy::SessionAccess;
        use project_orchestrator::chat::types::SpawnedBy;
        use project_orchestrator::neo4j::models::{
            ChatEventRecord, ChatSessionNode, ExecutionPlace,
        };

        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;

        let now = Utc::now();
        let parent = Uuid::new_v4();
        let child = Uuid::new_v4();
        let mk = |id: Uuid, spawned_by: Option<String>| ChatSessionNode {
            id,
            cli_session_id: None,
            project_slug: Some("p".to_string()),
            workspace_slug: None,
            cwd: "/tmp".to_string(),
            title: None,
            model: "sonnet".to_string(),
            created_at: now,
            updated_at: now,
            message_count: 0,
            total_cost_usd: None,
            conversation_id: None,
            preview: None,
            permission_mode: None,
            add_dirs: None,
            spawned_by,
            provider_id: None,
            routed_by: None,
            routing_mode: None,
            routing_pool: None,
            capabilities: None,
            engine: None,
            resume_token: None,
            execution_place: ExecutionPlace::Project,
            project_inferred: false,
            access: SessionAccess::Normal,
            owner: None,
            file_scope_lifted: false,
            mirror_of: None,
        };

        store.create_chat_session(&mk(parent, None)).await.unwrap();
        store
            .create_chat_session(&mk(
                child,
                Some(
                    SpawnedBy::Conversation {
                        parent_session_id: parent,
                        tool_use_id: None,
                    }
                    .to_json_string(),
                ),
            ))
            .await
            .unwrap();

        assert_eq!(
            store.get_chat_session(parent).await.unwrap().unwrap().model,
            "sonnet"
        );
        let (liste, total) = store
            .list_chat_sessions(None, None, 10, 0, true)
            .await
            .unwrap();
        assert_eq!(total, 2);
        assert_eq!(liste.len(), 2);

        // Lignée, portée par le JSON `spawned_by`.
        let enfants = store.get_session_children(parent).await.unwrap();
        assert_eq!(enfants.len(), 1);
        assert_eq!(enfants[0].id, child);
        let racine = store
            .get_session_root(&child.to_string())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(racine, parent.to_string());
        assert_eq!(
            store
                .get_session_tree(&parent.to_string())
                .await
                .unwrap()
                .len(),
            2
        );

        // Événements.
        let ev = ChatEventRecord {
            id: Uuid::new_v4(),
            session_id: parent,
            seq: 1,
            event_type: "user_message".to_string(),
            data: r#"{"content":"bonjour"}"#.to_string(),
            created_at: now,
        };
        store.store_chat_events(parent, vec![ev]).await.unwrap();
        assert_eq!(store.count_chat_events(parent).await.unwrap(), 1);
        assert_eq!(store.get_latest_chat_event_seq(parent).await.unwrap(), 1);
        assert_eq!(store.get_chat_events(parent, 0, 10).await.unwrap().len(), 1);
        assert_eq!(
            store
                .get_chat_events_paginated(parent, 0, 10)
                .await
                .unwrap()
                .len(),
            1
        );

        // Mise à jour partielle.
        let upd = store
            .update_chat_session(
                parent,
                None,
                Some("titre".to_string()),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(upd.title.as_deref(), Some("titre"));

        // Ancres.
        let ancre = store
            .add_anchor(
                parent,
                NewAnchor::new(
                    AnchorTargetType::File,
                    "src/main.rs",
                    [AnchorRole::Focus],
                    AnchorActor::User,
                    "moi",
                ),
            )
            .await
            .unwrap();
        assert_eq!(ancre.target_id, "src/main.rs");
        let jeu = store.get_anchor_set(parent).await.unwrap().unwrap();
        assert_eq!(jeu.version, 1);
        assert_eq!(jeu.anchors.len(), 1);
        let retrait = AnchorOp::Remove {
            anchor_id: ancre.id,
            expected_version: 1,
            by: AnchorActor::User,
            actor: "moi".to_string(),
        };
        // Un If-Match périmé refuse sans rien écrire.
        let conflit = store
            .apply_anchor_op_if(parent, retrait.clone(), Some(99))
            .await
            .unwrap_err();
        assert!(conflit.downcast_ref::<AnchorError>().is_some());
        assert_eq!(
            store.get_anchor_set(parent).await.unwrap().unwrap().version,
            1
        );
        // La bonne version passe.
        store
            .apply_anchor_op_if(parent, retrait, Some(1))
            .await
            .unwrap();
        assert_eq!(store.list_session_anchors(parent).await.unwrap().len(), 0);
        assert_eq!(
            store.get_anchor_set(parent).await.unwrap().unwrap().version,
            2
        );

        // Réglages du harness.
        store.put_llm_setting("global", "k", "v").await.unwrap();
        assert_eq!(
            store
                .get_llm_setting("global", "k")
                .await
                .unwrap()
                .as_deref(),
            Some("v")
        );
        assert_eq!(
            store.list_llm_settings("global", "").await.unwrap().len(),
            1
        );
        assert!(store.delete_llm_setting("global", "k").await.unwrap());
        assert_eq!(store.get_llm_setting("global", "k").await.unwrap(), None);

        // Réouverture : tout ce qui précède est durable.
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        assert_eq!(
            store
                .get_chat_session(parent)
                .await
                .unwrap()
                .unwrap()
                .title
                .as_deref(),
            Some("titre")
        );
        assert_eq!(store.count_chat_events(parent).await.unwrap(), 1);
        assert_eq!(store.get_session_children(parent).await.unwrap().len(), 1);

        // Suppression, et le cascade attendu.
        assert!(store.delete_chat_session(child).await.unwrap());
        assert!(store.delete_chat_session(parent).await.unwrap());
        assert_eq!(store.get_chat_session(parent).await.unwrap(), None);
        assert_eq!(
            store
                .list_chat_sessions(None, None, 10, 0, true)
                .await
                .unwrap()
                .1,
            0
        );
    }

    // ------------------------------------------------------------------
    // TriggerStore (`trigger.rs`) and PlanRunStore (`planrun.rs`)
    // ------------------------------------------------------------------

    fn a_plan(project_id: Option<Uuid>) -> project_orchestrator::neo4j::models::PlanNode {
        project_orchestrator::neo4j::models::PlanNode {
            id: Uuid::new_v4(),
            title: "plan of runs".to_string(),
            description: "d".to_string(),
            status: project_orchestrator::neo4j::models::PlanStatus::Approved,
            created_at: Utc::now(),
            created_by: "test".to_string(),
            priority: 0,
            project_id,
            execution_context: None,
            persona: None,
        }
    }

    fn a_trigger(plan_id: Uuid) -> project_orchestrator::runner::Trigger {
        project_orchestrator::runner::Trigger {
            id: Uuid::new_v4(),
            plan_id,
            trigger_type: project_orchestrator::runner::TriggerType::Webhook,
            config: serde_json::json!({"rule": {"event": "push"}}),
            enabled: true,
            cooldown_secs: 0,
            last_fired: None,
            fire_count: 0,
            created_at: Utc::now(),
            author: None,
            disabled_reason: None,
            approved_mark: None,
        }
    }

    /// Alerts (deduplicated, acknowledged, scoped by workspace), an event trigger
    /// and a lifecycle hook: written, closed, reopened, read back identical.
    #[tokio::test]
    async fn the_trigger_domain_survives_a_close_and_a_reopen() {
        use project_orchestrator::events::trigger::EventTrigger;
        use project_orchestrator::lifecycle::{LifecycleActionType, LifecycleHook, LifecycleScope};
        use project_orchestrator::neo4j::models::{AlertNode, AlertSeverity, WorkspaceNode};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let project = a_project("alerted");
        store.create_project(&project).await.unwrap();
        let ws = WorkspaceNode {
            id: Uuid::new_v4(),
            name: "ws".into(),
            slug: format!("ws-{}", Uuid::new_v4().simple()),
            description: None,
            created_at: Utc::now(),
            updated_at: None,
            metadata: serde_json::json!({}),
        };
        store.create_workspace(&ws).await.unwrap();
        store
            .add_project_to_workspace(ws.id, project.id)
            .await
            .unwrap();
        let alert = AlertNode::new_for_subject(
            "drift".into(),
            AlertSeverity::Critical,
            "3 commits behind".into(),
            Some(project.id),
            "drift",
        );
        store.create_alert(&alert).await.unwrap();
        let again = AlertNode::new_for_subject(
            "drift".into(),
            AlertSeverity::Critical,
            "4 commits behind".into(),
            Some(project.id),
            "drift",
        );
        store.create_alert(&again).await.unwrap();
        let global = AlertNode::new("other".into(), AlertSeverity::Info, "global".into(), None);
        store.create_alert(&global).await.unwrap();
        store.acknowledge_alert(global.id, "bob").await.unwrap();
        let event = EventTrigger {
            id: Uuid::new_v4(),
            name: "on sync".into(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: Some("project".into()),
            action_pattern: Some("synced".into()),
            payload_conditions: Some(serde_json::json!({"ok": true})),
            cooldown_secs: 60,
            enabled: true,
            project_scope: Some(project.id),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        store.create_event_trigger(&event).await.unwrap();
        let mut hook = LifecycleHook::new(
            "cascade".into(),
            LifecycleScope::Milestone,
            "closed".into(),
            LifecycleActionType::StartProtocol,
            serde_json::json!({"protocol": "x"}),
        );
        hook.project_id = Some(project.id);
        store.create_lifecycle_hook(&hook).await.unwrap();
        let before_alert = store.get_alert(alert.id).await.unwrap().unwrap();
        let before_event = store.get_event_trigger(event.id).await.unwrap().unwrap();
        let before_hook = store.get_lifecycle_hook(hook.id).await.unwrap().unwrap();
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        assert_eq!(
            serde_json::to_value(store.get_alert(alert.id).await.unwrap().unwrap()).unwrap(),
            serde_json::to_value(&before_alert).unwrap()
        );
        assert_eq!(before_alert.occurrence_count, 2);
        assert_eq!(before_alert.message, "4 commits behind");
        assert!(store.get_alert(again.id).await.unwrap().is_none());
        let (in_ws, total) = store
            .list_alerts(None, Some(&ws.slug), 10, 0)
            .await
            .unwrap();
        assert_eq!(
            (in_ws.len(), total),
            (1, 1),
            "the global alert is no project's"
        );
        assert_eq!(store.list_alerts(None, None, 10, 0).await.unwrap().1, 2);
        let pending = store.list_pending_alerts(None, 10).await.unwrap();
        assert_eq!(pending.len(), 1, "the acknowledged alert is not pending");
        assert_eq!(
            serde_json::to_value(store.get_event_trigger(event.id).await.unwrap().unwrap())
                .unwrap(),
            serde_json::to_value(&before_event).unwrap()
        );
        assert_eq!(
            serde_json::to_value(store.get_lifecycle_hook(hook.id).await.unwrap().unwrap())
                .unwrap(),
            serde_json::to_value(&before_hook).unwrap()
        );
        assert_eq!(
            store
                .list_hooks_for_scope(&LifecycleScope::Milestone, "closed", Some(project.id))
                .await
                .unwrap()
                .len(),
            1
        );
        store.close().await.unwrap();
    }

    /// A plan run, a trigger with its firings and a reserved signal, the third-party
    /// mark, the ingress journal and a delivery claim: written, closed, reopened,
    /// found as they were left.
    #[tokio::test]
    async fn the_plan_run_domain_survives_a_close_and_a_reopen() {
        use project_orchestrator::ingress::{ExternalEvent, IngressOutcome};
        use project_orchestrator::runner::{
            PlanContent, PlanRunStatus, RunnerState, SignalReservation, TriggerFiring,
            TriggerSource,
        };
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let plan = a_plan(None);
        store.create_plan(&plan).await.unwrap();
        let trigger = a_trigger(plan.id);
        store.create_trigger(&trigger).await.unwrap();
        let mut run = RunnerState::new(
            Uuid::new_v4(),
            plan.id,
            2,
            TriggerSource::Webhook {
                trigger_id: trigger.id,
                payload_hash: Some("h".into()),
            },
        );
        store.create_plan_run(&run).await.unwrap();
        run.current_wave = 1;
        run.cost_usd = 0.25;
        store.update_plan_run(&run).await.unwrap();
        let firing = TriggerFiring {
            id: Uuid::new_v4(),
            trigger_id: trigger.id,
            plan_run_id: Some(run.run_id),
            fired_at: Utc::now(),
            source_payload: Some(serde_json::json!({"ref": "main"})),
            start_error: None,
        };
        store.record_trigger_firing(&firing).await.unwrap();
        assert_eq!(
            store
                .reserve_trigger_signal(trigger.id, "github:d-1", 0)
                .await
                .unwrap(),
            SignalReservation::Reserved
        );
        store
            .mark_third_party_write(PlanContent::Plan(plan.id))
            .await
            .unwrap();
        let mark = store
            .plan_third_party_written_at(plan.id)
            .await
            .unwrap()
            .unwrap();
        let event = ExternalEvent {
            id: Uuid::new_v4(),
            trigger_id: trigger.id,
            source: "github".into(),
            outcome: IngressOutcome::Accepted,
            reason: None,
            delivery_id: Some("d-1".into()),
            event_type: Some("push".into()),
            action: None,
            body_bytes: 12,
            received_at: Utc::now(),
        };
        store.record_external_event(&event).await.unwrap();
        assert!(store
            .claim_external_delivery(trigger.id, "github:d-1")
            .await
            .unwrap());
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        let found = store.get_plan_run(run.run_id).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&found).unwrap(),
            serde_json::to_value(&run).unwrap(),
            "the run reads back as written, state_json included"
        );
        assert_eq!(found.status, PlanRunStatus::Running);
        assert_eq!(store.list_active_plan_runs().await.unwrap().len(), 1);
        let t = store.get_trigger(trigger.id).await.unwrap().unwrap();
        assert_eq!(t.fire_count, 1);
        assert!(t.last_fired.is_some(), "the reservation moved last_fired");
        assert_eq!(
            store
                .reserve_trigger_signal(trigger.id, "github:d-1", 0)
                .await
                .unwrap(),
            SignalReservation::Duplicate,
            "the reserved signal survived the reopen"
        );
        let firings = store.list_trigger_firings(trigger.id, 10).await.unwrap();
        assert_eq!(firings.len(), 1);
        assert_eq!(firings[0].plan_run_id, Some(run.run_id));
        assert_eq!(
            store.plan_third_party_written_at(plan.id).await.unwrap(),
            Some(mark)
        );
        assert_eq!(
            store.list_external_events(trigger.id, 10).await.unwrap(),
            vec![event]
        );
        assert!(
            !store
                .claim_external_delivery(trigger.id, "github:d-1")
                .await
                .unwrap(),
            "the claim survived the reopen"
        );
        store.close().await.unwrap();
    }

    /// A creation interrupted at each of its steps (a hook: 13 properties, so 16
    /// steps; an external event: 10 properties, 13 steps): once reopened, the draft
    /// is gone (counted, except before the node exists), nothing is quarantined,
    /// and the same creation goes through.
    #[tokio::test]
    async fn an_interrupted_hook_or_event_creation_reopens_to_a_consistent_base() {
        use project_orchestrator::ingress::{ExternalEvent, IngressOutcome};
        use project_orchestrator::lifecycle::{LifecycleActionType, LifecycleHook, LifecycleScope};
        let dir = tempfile::tempdir().unwrap();
        let points = (1..=16)
            .map(|k| ("hook", k))
            .chain((1..=13).map(|k| ("event", k)));
        for (what, k) in points {
            let base = dir.path().join(format!("{what}-{k}"));
            let store = open(&base).await;
            let hook = LifecycleHook::new(
                "h".into(),
                LifecycleScope::Task,
                "completed".into(),
                LifecycleActionType::EmitAlert,
                serde_json::json!({}),
            );
            let event = ExternalEvent {
                id: Uuid::new_v4(),
                trigger_id: Uuid::new_v4(),
                source: "github".into(),
                outcome: IngressOutcome::Rejected,
                reason: Some("invalid_signature".into()),
                delivery_id: None,
                event_type: None,
                action: None,
                body_bytes: 0,
                received_at: Utc::now(),
            };
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            let err = if what == "hook" {
                store.create_lifecycle_hook(&hook).await.unwrap_err()
            } else {
                store.record_external_event(&event).await.unwrap_err()
            };
            assert!(
                format!("{err:#}").contains("interruption injectée"),
                "{what} {k}: {err:#}"
            );
            assert!(store.get_lifecycle_hook(hook.id).await.unwrap().is_none());
            assert!(store
                .list_external_events(event.trigger_id, 10)
                .await
                .unwrap()
                .is_empty());
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert!(rapport.en_quarantaine.is_empty(), "{what} {k}: {rapport:?}");
            assert_eq!(
                rapport.brouillons_supprimes,
                usize::from(k > 1),
                "{what} {k}: {rapport:?}"
            );
            assert!(store.get_lifecycle_hook(hook.id).await.unwrap().is_none());
            assert!(store
                .list_external_events(event.trigger_id, 10)
                .await
                .unwrap()
                .is_empty());
            store.create_lifecycle_hook(&hook).await.unwrap();
            store.record_external_event(&event).await.unwrap();
            store.close().await.unwrap();
        }
    }

    /// A base written by someone else, with nodes this backend cannot decode: each
    /// is quarantined (named, untouched, never served), the rest is served.
    #[tokio::test]
    async fn unreadable_trigger_and_run_nodes_of_a_foreign_base_are_quarantined() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (alert, run, hook, good) = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
            Uuid::new_v4(),
        );
        write_foreign(&base, move |db| {
            let s = |x: &str| Value::from(x.to_string());
            // A severity Neo4j cannot parse either.
            db.create_node_with_props(
                &["Alert"],
                vec![
                    ("id", s(&alert.to_string())),
                    ("alert_type", s("t")),
                    ("severity", s("apocalyptic")),
                    ("message", s("m")),
                    ("created_at", s("2026-01-01T00:00:00Z")),
                    ("dedup_key", s("k")),
                ],
            );
            // A run whose state_json is not JSON: Neo4j falls back to the flat fields.
            db.create_node_with_props(
                &["PlanRun"],
                vec![
                    ("run_id", s(&run.to_string())),
                    ("plan_id", s(&Uuid::new_v4().to_string())),
                    ("status", s("running")),
                    ("started_at", s("2026-01-01T00:00:00Z")),
                    ("state_json", s("{tronqué")),
                ],
            );
            // A hook whose scope Neo4j reads as `Task` in silence.
            db.create_node_with_props(
                &["LifecycleHook"],
                vec![
                    ("id", s(&hook.to_string())),
                    ("name", s("h")),
                    ("scope", s("Galaxy")),
                    ("on_status", s("done")),
                    ("action_type", s("EmitAlert")),
                    ("priority", Value::from(1_i64)),
                    ("created_at", s("2026-01-01T00:00:00Z")),
                ],
            );
            // A readable legacy run (no state_json): served from its flat fields.
            db.create_node_with_props(
                &["PlanRun"],
                vec![
                    ("run_id", s(&good.to_string())),
                    ("plan_id", s(&Uuid::new_v4().to_string())),
                    ("status", s("completed")),
                    ("started_at", s("2026-01-01T00:00:00Z")),
                    ("completed_tasks", s("")),
                ],
            );
        })
        .await;
        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert_eq!(rapport.en_quarantaine.len(), 3, "{rapport:?}");
        for needle in ["apocalyptic", "state_json", "Galaxy"] {
            assert!(
                rapport.en_quarantaine.iter().any(|q| q.contains(needle)),
                "{needle}: {rapport:?}"
            );
        }
        assert!(store.get_alert(alert).await.unwrap().is_none());
        assert!(store.get_plan_run(run).await.unwrap().is_none());
        assert!(store.get_lifecycle_hook(hook).await.unwrap().is_none());
        let legacy = store.get_plan_run(good).await.unwrap().unwrap();
        assert_eq!(
            legacy.status,
            project_orchestrator::runner::PlanRunStatus::Completed
        );
        store.close().await.unwrap();
    }

    /// Every method of `PlanRunStore` listed as refused refuses BY NAME, and the
    /// one partial refusal (`mark_third_party_write` of a task) too.
    #[tokio::test]
    async fn plan_run_refusals_are_named() {
        // `CostGroupBy` is not reachable from outside the crate: the ninth refusal,
        // `agent_execution_cost_breakdown`, is checked by the unit test of `planrun.rs`.
        use project_orchestrator::neo4j::AgentExecutionNode;
        use project_orchestrator::runner::PlanContent;
        let store = temporary(Durabilite::Wal).await;
        let id = Uuid::new_v4();
        let ae = AgentExecutionNode::new(id, id);
        let names: Vec<NonPorte> = vec![
            not_ported(store.create_agent_execution(&ae).await),
            not_ported(store.update_agent_execution(&ae).await),
            not_ported(store.get_agent_executions_for_run(id).await),
            not_ported(store.record_agent_execution_verification(id, "{}").await),
            not_ported(store.list_running_agent_executions().await),
            not_ported(store.get_all_pagerank_values(id).await),
            not_ported(store.get_community_risk_vectors(id).await),
            not_ported(store.get_all_risk_score_values(id).await),
        ];
        let expected: Vec<NonPorte> = refusees_planrun()
            .iter()
            .filter(|m| **m != "PlanRunStore::agent_execution_cost_breakdown")
            .map(|m| NonPorte(m))
            .collect();
        assert_eq!(refusees_planrun().len(), 9);
        assert_eq!(names, expected);
        for content in [
            PlanContent::Task(id),
            PlanContent::Step(id),
            PlanContent::Constraint(id),
            PlanContent::Decision(id),
        ] {
            let e = store.mark_third_party_write(content).await.unwrap_err();
            assert_eq!(
                e.downcast_ref::<NonPorte>(),
                Some(&NonPorte("PlanRunStore::mark_third_party_write")),
                "{content:?}"
            );
        }
    }

    /// The guards under contention: eight concurrent reservations of one signal,
    /// one wins; eight concurrent observations of one alert condition, one node
    /// counting eight.
    #[tokio::test]
    async fn trigger_guards_hold_under_concurrency() {
        use futures::future::join_all;
        use project_orchestrator::neo4j::models::{AlertNode, AlertSeverity};
        use project_orchestrator::runner::SignalReservation;
        let store = temporary(Durabilite::Wal).await;
        let plan = a_plan(None);
        store.create_plan(&plan).await.unwrap();
        let trigger = a_trigger(plan.id);
        store.create_trigger(&trigger).await.unwrap();
        let outcomes =
            join_all((0..8).map(|_| store.reserve_trigger_signal(trigger.id, "race", 0))).await;
        assert_eq!(
            outcomes
                .iter()
                .filter(|o| matches!(o, Ok(SignalReservation::Reserved)))
                .count(),
            1,
            "{outcomes:?}"
        );
        let alerts: Vec<AlertNode> = (0..8)
            .map(|i| {
                AlertNode::new_for_subject(
                    "race".into(),
                    AlertSeverity::Warning,
                    format!("seen {i}"),
                    None,
                    "one-condition",
                )
            })
            .collect();
        let outcomes = join_all(alerts.iter().map(|a| store.create_alert(a))).await;
        assert!(outcomes.iter().all(|o| o.is_ok()), "{outcomes:?}");
        let (all, total) = store.list_alerts(None, None, 10, 0).await.unwrap();
        assert_eq!(total, 1);
        assert_eq!(all[0].occurrence_count, 8);
    }

    // ------------------------------------------------------------------
    // Review of #1208: quarantined keys stay reserved, the cooldown race,
    // the repair of `last_fired`, `enabled` absent, the sort of an alert
    // without a stored priority.
    // ------------------------------------------------------------------

    /// A delivery, a signal, a run id, an alert `dedup_key` and an event-trigger id
    /// whose nodes are on disk but unreadable stay RESERVED: claiming, reserving or
    /// creating them again is refused by name (never `true`, `Reserved`, a new node).
    #[tokio::test]
    async fn quarantined_keys_stay_reserved_and_are_refused_by_name() {
        use project_orchestrator::neo4j::models::{AlertNode, AlertSeverity};
        use project_orchestrator::runner::{RunnerState, TriggerSource};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (plan, trigger, run, event_trigger) = {
            let store = open(&base).await;
            let plan = a_plan(None);
            store.create_plan(&plan).await.unwrap();
            let trigger = a_trigger(plan.id);
            store.create_trigger(&trigger).await.unwrap();
            store.close().await.unwrap();
            (plan, trigger, Uuid::new_v4(), Uuid::new_v4())
        };
        let t = trigger.id.to_string();
        let (r, et) = (run.to_string(), event_trigger.to_string());
        write_foreign(&base, move |db| {
            let s = |x: &str| Value::from(x.to_string());
            // A claimed delivery and a reserved signal whose date is unreadable.
            db.create_node_with_props(
                &["ExternalDelivery"],
                vec![
                    ("trigger_id", s(&t)),
                    ("delivery_key", s("github:d-1")),
                    ("claimed_at", s("pas une date")),
                ],
            );
            db.create_node_with_props(
                &["TriggerSignal"],
                vec![
                    ("trigger_id", s(&t)),
                    ("key", s("github:d-1")),
                    ("reserved_at", s("pas une date")),
                ],
            );
            // A run whose state_json is not JSON.
            db.create_node_with_props(
                &["PlanRun"],
                vec![("run_id", s(&r)), ("state_json", s("{tronqué"))],
            );
            // An alert of an unknown severity, and an event trigger with no name.
            db.create_node_with_props(
                &["Alert"],
                vec![
                    ("id", s(&Uuid::new_v4().to_string())),
                    ("dedup_key", s("drift:global:x")),
                    ("severity", s("apocalyptic")),
                ],
            );
            db.create_node_with_props(&["EventTrigger"], vec![("id", s(&et))]);
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 5);
        let reserved = |r: anyhow::Result<String>| {
            let e = r.expect_err("a reserved key must be refused");
            assert!(format!("{e:#}").contains("réservé"), "{e:#}");
        };
        reserved(
            store
                .claim_external_delivery(trigger.id, "github:d-1")
                .await
                .map(|b| format!("{b}")),
        );
        reserved(
            store
                .release_external_delivery(trigger.id, "github:d-1")
                .await
                .map(|b| format!("{b}")),
        );
        reserved(
            store
                .reserve_trigger_signal(trigger.id, "github:d-1", 0)
                .await
                .map(|r| format!("{r:?}")),
        );
        reserved(
            store
                .create_plan_run(&RunnerState::new(run, plan.id, 1, TriggerSource::Manual))
                .await
                .map(|_| String::new()),
        );
        let mut alert = AlertNode::new("drift".into(), AlertSeverity::Warning, "x".into(), None);
        alert.dedup_key = "drift:global:x".into();
        reserved(store.create_alert(&alert).await.map(|_| String::new()));
        let mut et = project_orchestrator::events::trigger::EventTrigger {
            id: event_trigger,
            name: "again".into(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: None,
            action_pattern: None,
            payload_conditions: None,
            cooldown_secs: 0,
            enabled: true,
            project_scope: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        reserved(store.create_event_trigger(&et).await.map(|u| u.to_string()));
        // What is not reserved is still served.
        et.id = Uuid::new_v4();
        store.create_event_trigger(&et).await.unwrap();
        assert!(store
            .claim_external_delivery(trigger.id, "github:d-2")
            .await
            .unwrap());
        assert_eq!(store.list_alerts(None, None, 10, 0).await.unwrap().1, 0);
        store.close().await.unwrap();
    }

    /// The cooldown under contention, on a FRESH trigger: eight distinct keys at
    /// once under a one-hour cooldown, exactly one is `Reserved`, seven `Cooldown`.
    #[tokio::test]
    async fn the_cooldown_holds_for_eight_concurrent_distinct_signals() {
        use futures::future::join_all;
        use project_orchestrator::runner::SignalReservation;
        let store = temporary(Durabilite::Wal).await;
        let plan = a_plan(None);
        store.create_plan(&plan).await.unwrap();
        let trigger = a_trigger(plan.id);
        store.create_trigger(&trigger).await.unwrap();
        let keys: Vec<String> = (0..8).map(|i| format!("github:d-{i}")).collect();
        let outcomes = join_all(
            keys.iter()
                .map(|k| store.reserve_trigger_signal(trigger.id, k, 3600)),
        )
        .await;
        let count = |r: SignalReservation| {
            outcomes
                .iter()
                .filter(|o| matches!(o, Ok(x) if *x == r))
                .count()
        };
        assert_eq!(
            (
                count(SignalReservation::Reserved),
                count(SignalReservation::Cooldown)
            ),
            (1, 7),
            "{outcomes:?}"
        );
    }

    /// A reservation interrupted after its signal is committed and before
    /// `last_fired` moves: at reopen `last_fired` is repaired to the signal, so the
    /// cooldown holds and the key stays taken.
    #[tokio::test]
    async fn last_fired_is_repaired_to_the_latest_signal_at_reopen() {
        use project_orchestrator::runner::SignalReservation;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let plan = a_plan(None);
        store.create_plan(&plan).await.unwrap();
        let trigger = a_trigger(plan.id);
        store.create_trigger(&trigger).await.unwrap();
        // The signal node: before the node, before the marker, three properties,
        // before the marker goes (6 steps); step 7 is `last_fired`.
        store.interrompre_au_prochain(Interruption::AvantEtape(7));
        assert!(store
            .reserve_trigger_signal(trigger.id, "k", 3600)
            .await
            .is_err());
        assert!(store
            .get_trigger(trigger.id)
            .await
            .unwrap()
            .unwrap()
            .last_fired
            .is_none());
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture(), &RapportOuverture::default());
        let fired = store
            .get_trigger(trigger.id)
            .await
            .unwrap()
            .unwrap()
            .last_fired
            .expect("last_fired repaired to the committed signal");
        assert!((Utc::now() - fired).num_seconds() < 600, "{fired}");
        assert_eq!(
            store
                .reserve_trigger_signal(trigger.id, "other", 3600)
                .await
                .unwrap(),
            SignalReservation::Cooldown
        );
        assert_eq!(
            store
                .reserve_trigger_signal(trigger.id, "k", 0)
                .await
                .unwrap(),
            SignalReservation::Duplicate
        );
        store.close().await.unwrap();
    }

    /// A hook or an event trigger without `enabled` (a foreign or old node) is not
    /// enabled: Neo4j's `enabled = true` filters exclude it, and a hook is never
    /// run on a guess.
    #[tokio::test]
    async fn a_hook_or_event_trigger_without_enabled_is_not_run() {
        use project_orchestrator::lifecycle::LifecycleScope;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let (hook, trigger) = (Uuid::new_v4(), Uuid::new_v4());
        write_foreign(&base, move |db| {
            let s = |x: &str| Value::from(x.to_string());
            db.create_node_with_props(
                &["LifecycleHook"],
                vec![
                    ("id", s(&hook.to_string())),
                    ("name", s("old hook")),
                    ("scope", s("Task")),
                    ("on_status", s("completed")),
                    ("action_type", s("EmitAlert")),
                    ("action_config", s("{}")),
                    ("priority", Value::from(1_i64)),
                    ("project_id", s("")),
                    ("created_at", s("2026-01-01T00:00:00Z")),
                ],
            );
            db.create_node_with_props(
                &["EventTrigger"],
                vec![
                    ("id", s(&trigger.to_string())),
                    ("name", s("old trigger")),
                    ("protocol_id", s(&Uuid::new_v4().to_string())),
                    ("created_at", s("2026-01-01T00:00:00Z")),
                    ("updated_at", s("2026-01-01T00:00:00Z")),
                ],
            );
        })
        .await;
        let store = open(&base).await;
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        assert!(store
            .list_hooks_for_scope(&LifecycleScope::Task, "completed", None)
            .await
            .unwrap()
            .is_empty());
        assert!(store
            .list_event_triggers(None, true)
            .await
            .unwrap()
            .is_empty());
        assert!(
            !store
                .get_lifecycle_hook(hook)
                .await
                .unwrap()
                .unwrap()
                .enabled
        );
        store.close().await.unwrap();
    }

    /// An alert read without a stored `priority` sorts as 0 (`coalesce(priority,
    /// 0)`), behind an alert of a lower stored priority.
    #[tokio::test]
    async fn an_alert_without_a_stored_priority_sorts_as_zero() {
        use project_orchestrator::neo4j::models::{AlertNode, AlertSeverity};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let old = Uuid::new_v4();
        write_foreign(&base, move |db| {
            let s = |x: &str| Value::from(x.to_string());
            // Critical, recent, no priority: it would compute to ~1.0.
            let now = Utc::now().to_rfc3339();
            db.create_node_with_props(
                &["Alert"],
                vec![
                    ("id", s(&old.to_string())),
                    ("alert_type", s("old")),
                    ("severity", s("critical")),
                    ("message", s("m")),
                    ("acknowledged", Value::from(false)),
                    ("created_at", s(&now)),
                    ("dedup_key", s("old:global:m")),
                ],
            );
        })
        .await;
        let store = open(&base).await;
        let info = AlertNode::new("new".into(), AlertSeverity::Info, "i".into(), None);
        store.create_alert(&info).await.unwrap();
        let ids: Vec<Uuid> = store
            .list_pending_alerts(None, 10)
            .await
            .unwrap()
            .iter()
            .map(|a| a.id)
            .collect();
        assert_eq!(
            ids,
            vec![info.id, old],
            "the stored 0.2 before the coalesced 0"
        );
        store.close().await.unwrap();
    }

    /// A long value (> 256 bytes: a run's `state_json`, a trigger's `config`, an
    /// event payload, an alert message, an external event's reason) goes to a blob
    /// column the pinned WAL does not recover: the write pays the FULL flush before
    /// it is acknowledged, and so does a write that REPLACES such a value. Counted
    /// with obrain's own fsync oracle, without any `close()`.
    #[tokio::test]
    async fn long_values_of_triggers_and_runs_pay_the_full_flush() {
        use project_orchestrator::events::trigger::EventTrigger;
        use project_orchestrator::ingress::{ExternalEvent, IngressOutcome};
        use project_orchestrator::neo4j::models::{AlertNode, AlertSeverity};
        use project_orchestrator::runner::{RunnerState, TriggerSource};
        let store = temporary(Durabilite::Wal).await;
        let fsyncs = || store.fsyncs_du_wal().unwrap();
        let plan = a_plan(None);
        store.create_plan(&plan).await.unwrap();
        let long = "x".repeat(300);

        // The cost of a write with nothing long.
        let t0 = fsyncs();
        assert!(store
            .claim_external_delivery(Uuid::new_v4(), "short")
            .await
            .unwrap());
        let small = fsyncs() - t0;
        assert!(small >= 1, "the WAL fsync ran");
        let cost = |before: u64, what: &str| {
            let c = fsyncs() - before;
            assert!(
                c > small,
                "{what}: full flush expected ({c} fsyncs against {small})"
            );
        };

        let mut run = RunnerState::new(Uuid::new_v4(), plan.id, 1, TriggerSource::Manual);
        assert!(serde_json::to_string(&run).unwrap().len() > 256);
        let t = fsyncs();
        store.create_plan_run(&run).await.unwrap();
        cost(t, "create_plan_run (state_json)");
        run.current_wave = 1;
        let t = fsyncs();
        store.update_plan_run(&run).await.unwrap();
        cost(t, "update_plan_run (state_json)");

        let mut trigger = a_trigger(plan.id);
        trigger.config = serde_json::json!({ "rule": long });
        let t = fsyncs();
        store.create_trigger(&trigger).await.unwrap();
        cost(t, "create_trigger (config)");

        let mut event_trigger = EventTrigger {
            id: Uuid::new_v4(),
            name: "e".into(),
            protocol_id: Uuid::new_v4(),
            entity_type_pattern: None,
            action_pattern: None,
            payload_conditions: Some(serde_json::json!({ "k": long })),
            cooldown_secs: 0,
            enabled: true,
            project_scope: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let t = fsyncs();
        store.create_event_trigger(&event_trigger).await.unwrap();
        cost(t, "create_event_trigger (payload_conditions)");
        // The long value REPLACED by a short one: its old blob is rewritten too.
        let t = fsyncs();
        assert!(store
            .update_event_trigger(
                event_trigger.id,
                None,
                None,
                None,
                None,
                Some(None),
                None,
                None
            )
            .await
            .unwrap());
        cost(t, "update_event_trigger replacing a long payload");
        event_trigger.id = Uuid::new_v4();

        let alert = AlertNode::new("long".into(), AlertSeverity::Info, long.clone(), None);
        let t = fsyncs();
        store.create_alert(&alert).await.unwrap();
        cost(t, "create_alert (message)");

        let event = ExternalEvent {
            id: Uuid::new_v4(),
            trigger_id: trigger.id,
            source: "github".into(),
            outcome: IngressOutcome::Rejected,
            reason: Some(long.clone()),
            delivery_id: None,
            event_type: None,
            action: None,
            body_bytes: 0,
            received_at: Utc::now(),
        };
        let t = fsyncs();
        store.record_external_event(&event).await.unwrap();
        cost(t, "record_external_event (reason)");
    }

    // ------------------------------------------------------------------
    // Persona and feature graph (persona.rs, featuregraph.rs)
    // ------------------------------------------------------------------

    fn a_persona(
        project: Option<Uuid>,
        name: &str,
        energy: f64,
    ) -> project_orchestrator::neo4j::models::PersonaNode {
        use project_orchestrator::neo4j::models::{PersonaNode, PersonaOrigin, PersonaStatus};
        PersonaNode {
            id: Uuid::new_v4(),
            project_id: project,
            name: name.to_string(),
            description: format!("{name}, experte"),
            status: PersonaStatus::Active,
            complexity_default: None,
            timeout_secs: Some(300),
            max_cost_usd: None,
            model_preference: Some("opus".into()),
            system_prompt_override: None,
            energy,
            cohesion: 0.0,
            activation_count: 0,
            success_rate: 0.0,
            avg_duration_secs: 0.0,
            last_activated: None,
            energy_boost_accumulated: 0.0,
            energy_history: vec![],
            origin: PersonaOrigin::Manual,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    fn a_feature_graph(
        project: Uuid,
        name: &str,
    ) -> project_orchestrator::neo4j::models::FeatureGraphNode {
        project_orchestrator::neo4j::models::FeatureGraphNode {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: Some(format!("{name} feature")),
            project_id: project,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            entity_count: None,
            entry_function: Some("main".into()),
            build_depth: Some(2),
            include_relations: Some(vec!["CALLS".into()]),
        }
    }

    /// Personas, their edges and feature graphs written, the base closed and reopened:
    /// everything reads back the same (the indexes are rebuilt from the base).
    #[tokio::test]
    async fn personas_and_feature_graphs_survive_a_reopen() {
        use project_orchestrator::neo4j::models::PersonaStatus;
        use project_orchestrator::notes::{Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("personas");
        store.create_project(&p).await.unwrap();
        let fg = a_feature_graph(p.id, "auth");
        store.create_feature_graph(&fg).await.unwrap();
        let parent = a_persona(None, "parent", 0.8);
        let mut child = a_persona(Some(p.id), "child", 0.6);
        store.create_persona(&parent).await.unwrap();
        store.create_persona(&child).await.unwrap();
        let n = Note::new(
            Some(p.id),
            NoteType::Tip,
            "use the vault".into(),
            "alice".into(),
        );
        store.create_note(&n).await.unwrap();
        store.add_persona_note(child.id, n.id, 0.9).await.unwrap();
        store
            .add_persona_extends(child.id, parent.id)
            .await
            .unwrap();
        store
            .set_persona_feature_graph(child.id, fg.id)
            .await
            .unwrap();
        child.status = PersonaStatus::Dormant;
        child.name = "child-renamed".into();
        store.update_persona(&child).await.unwrap();
        store.increment_persona_activation(child.id).await.unwrap();
        let before = store.get_persona_subgraph(child.id).await.unwrap();
        let fg_before = store.get_feature_graph(fg.id).await.unwrap().unwrap();
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert_eq!(rapport, RapportOuverture::default(), "{rapport:?}");
        let relue = store.get_persona(child.id).await.unwrap().unwrap();
        assert_eq!(relue.name, "child-renamed");
        assert_eq!(relue.status, PersonaStatus::Dormant);
        assert_eq!(relue.activation_count, 1);
        assert_eq!(relue.model_preference.as_deref(), Some("opus"));
        assert_eq!(relue.project_id, Some(p.id));
        let (listed, total) = store
            .list_personas(p.id, Some(PersonaStatus::Dormant), 10, 0)
            .await
            .unwrap();
        assert_eq!((listed.len(), total), (1, 1));
        assert_eq!(store.list_global_personas().await.unwrap()[0].id, parent.id);
        let after = store.get_persona_subgraph(child.id).await.unwrap();
        assert_eq!(
            serde_json::to_value(&after).unwrap(),
            serde_json::to_value(&before).unwrap()
        );
        assert_eq!(after.notes[0].weight, 0.9);
        assert_eq!(after.feature_graph_id, Some(fg.id));
        assert_eq!(after.parents[0].entity_id, parent.id.to_string());
        let fg_after = store.get_feature_graph(fg.id).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&fg_after).unwrap(),
            serde_json::to_value(&fg_before).unwrap()
        );
        let listed = store.list_feature_graphs(Some(p.id)).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].entity_count, Some(0));

        // Removing the edges, then the nodes, is durable too.
        store.remove_persona_note(child.id, n.id).await.unwrap();
        store
            .remove_persona_extends(child.id, parent.id)
            .await
            .unwrap();
        store.remove_persona_feature_graph(child.id).await.unwrap();
        assert!(store.delete_feature_graph(fg.id).await.unwrap());
        assert!(store.delete_persona(parent.id).await.unwrap());
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        let sg = store.get_persona_subgraph(child.id).await.unwrap();
        assert!(sg.notes.is_empty() && sg.parents.is_empty() && sg.feature_graph_id.is_none());
        assert!(store.get_persona(parent.id).await.unwrap().is_none());
        assert!(store.get_feature_graph(fg.id).await.unwrap().is_none());
    }

    /// A persona creation, and a feature-graph creation, interrupted at every step
    /// reopen to a consistent base: the draft is deleted with its edges, or the node
    /// is whole when the commit point was passed.
    #[tokio::test]
    async fn an_interrupted_persona_or_feature_graph_creation_reopens_consistently() {
        let dir = tempfile::tempdir().unwrap();
        let mut points = vec![Interruption::ApresLeNoeud, Interruption::ApresLeMarqueur];
        points.extend((1..=25).map(Interruption::ApresLaPropriete));
        points.extend([
            Interruption::AvantLeRetraitDuMarqueur,
            Interruption::ApresLeRetraitDuMarqueur,
        ]);
        let mut touches = 0;
        for (k, point) in points.iter().copied().enumerate() {
            for graphe in [false, true] {
                let base = dir.path().join(format!("p-{k}-{graphe}"));
                let store = open(&base).await;
                let p = a_project("interrupted");
                store.create_project(&p).await.unwrap();
                let persona = a_persona(Some(p.id), "draft", 0.5);
                let fg = a_feature_graph(p.id, "draft");
                store.interrompre_au_prochain(point);
                let issue = if graphe {
                    store.create_feature_graph(&fg).await
                } else {
                    store.create_persona(&persona).await
                };
                if issue.is_ok() {
                    store.close().await.unwrap();
                    continue;
                }
                touches += 1;
                assert!(store.get_persona(persona.id).await.unwrap().is_none());
                assert!(store.get_feature_graph(fg.id).await.unwrap().is_none());
                store.close().await.unwrap();
                drop(store);

                let store = open(&base).await;
                let rapport = store.rapport_d_ouverture().clone();
                assert!(rapport.en_quarantaine.is_empty(), "{point:?}: {rapport:?}");
                let present = if graphe {
                    store.get_feature_graph(fg.id).await.unwrap().is_some()
                } else {
                    store.get_persona(persona.id).await.unwrap().is_some()
                };
                if point == Interruption::ApresLeRetraitDuMarqueur {
                    assert!(present, "{point:?}");
                } else {
                    assert_eq!(rapport.brouillons_supprimes, 1, "{point:?}: {rapport:?}");
                    assert!(!present, "{point:?}");
                    // The draft and its BELONGS_TO edge are gone: the id is free again.
                    if graphe {
                        store.create_feature_graph(&fg).await.unwrap();
                    } else {
                        store.create_persona(&persona).await.unwrap();
                    }
                }
                store.close().await.unwrap();
            }
        }
        // Persona: 16 properties written here; feature graph: 9.
        assert!(touches >= 30, "the points were reached: {touches}");
    }

    /// A foreign base: unreadable Persona and FeatureGraph nodes are quarantined (named,
    /// left untouched, their ids reserved); a persona that KNOWS a file and a graph that
    /// INCLUDES an entity are served as nodes, and their code-dependent views refuse by name.
    #[tokio::test]
    async fn foreign_persona_and_feature_graph_nodes_are_quarantined_or_refused() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("foreign");
        store.create_project(&p).await.unwrap();
        let knows = a_persona(Some(p.id), "knows-a-file", 0.5);
        store.create_persona(&knows).await.unwrap();
        let fg = a_feature_graph(p.id, "includes");
        store.create_feature_graph(&fg).await.unwrap();
        store.close().await.unwrap();
        drop(store);

        let (bad_persona, bad_graph) = (Uuid::new_v4(), Uuid::new_v4());
        let (pid, kid, fid) = (p.id, knows.id, fg.id);
        write_foreign(&base, move |db| {
            // A persona with no `created_at`, a graph whose `created_at` does not parse.
            db.create_node_with_props(
                &["Persona"],
                vec![
                    ("id", Value::from(bad_persona.to_string())),
                    ("name", Value::from("sans date")),
                ],
            );
            db.create_node_with_props(
                &["FeatureGraph"],
                vec![
                    ("id", Value::from(bad_graph.to_string())),
                    ("name", Value::from("mauvaise date")),
                    ("project_id", Value::from(pid.to_string())),
                    ("created_at", Value::from("hier")),
                    ("updated_at", Value::from("hier")),
                ],
            );
            let file = db.create_node(&["File"]);
            db.set_node_property(file, "path", Value::from("/src/lib.rs"));
            let pn = db.find_nodes_by_property("id", &Value::from(kid.to_string()))[0];
            db.create_edge_with_props(pn, file, "KNOWS", [("weight", Value::from(0.8))]);
            let gn = db.find_nodes_by_property("id", &Value::from(fid.to_string()))[0];
            db.create_edge(gn, file, "INCLUDES_ENTITY");
        })
        .await;

        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert_eq!(rapport.en_quarantaine.len(), 2, "{rapport:?}");
        assert_eq!(rapport.brouillons_supprimes, 0, "{rapport:?}");
        assert!(store.get_persona(bad_persona).await.unwrap().is_none());
        assert!(store.get_feature_graph(bad_graph).await.unwrap().is_none());
        // Their ids are reserved: no second node with the same id.
        let mut again = a_persona(None, "reuse", 0.1);
        again.id = bad_persona;
        let err = store.create_persona(&again).await.unwrap_err();
        assert!(format!("{err:#}").contains("réservé"), "{err:#}");
        let mut again = a_feature_graph(p.id, "reuse");
        again.id = bad_graph;
        let err = store.create_feature_graph(&again).await.unwrap_err();
        assert!(format!("{err:#}").contains("réservé"), "{err:#}");

        // The nodes are served; what would need the code graph refuses by name.
        assert!(store.get_persona(knows.id).await.unwrap().is_some());
        assert_eq!(
            not_ported(store.get_persona_subgraph(knows.id).await),
            NonPorte("PersonaStore::get_persona_subgraph")
        );
        assert_eq!(
            store.list_feature_graphs(Some(p.id)).await.unwrap()[0].entity_count,
            Some(1)
        );
        assert_eq!(
            not_ported(store.get_feature_graph_detail(fg.id).await),
            NonPorte("FeatureGraphStore::get_feature_graph_detail")
        );
        // Deleting detaches every edge, the foreign ones included.
        assert!(store.delete_persona(knows.id).await.unwrap());
        assert!(store.delete_feature_graph(fg.id).await.unwrap());
    }

    /// Every method of `persona::REFUSEES` and `featuregraph::REFUSEES` refuses by its
    /// own name, and writes nothing.
    #[tokio::test]
    async fn persona_and_feature_graph_refusals_are_named() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("refus");
        store.create_project(&p).await.unwrap();
        let a = a_persona(Some(p.id), "a", 0.5);
        store.create_persona(&a).await.unwrap();
        let fg = a_feature_graph(p.id, "fg");
        store.create_feature_graph(&fg).await.unwrap();
        let (id, x) = (a.id, Uuid::new_v4());
        let cas: Vec<(NonPorte, &str)> = vec![
            (
                not_ported(store.add_persona_skill(id, x).await),
                "PersonaStore::add_persona_skill",
            ),
            (
                not_ported(store.remove_persona_skill(id, x).await),
                "PersonaStore::remove_persona_skill",
            ),
            (
                not_ported(store.add_persona_protocol(id, x).await),
                "PersonaStore::add_persona_protocol",
            ),
            (
                not_ported(store.remove_persona_protocol(id, x).await),
                "PersonaStore::remove_persona_protocol",
            ),
            (
                not_ported(store.add_persona_file(id, "src/a.rs", 0.5).await),
                "PersonaStore::add_persona_file",
            ),
            (
                not_ported(store.remove_persona_file(id, "src/a.rs").await),
                "PersonaStore::remove_persona_file",
            ),
            (
                not_ported(store.add_persona_function(id, "f", 0.5).await),
                "PersonaStore::add_persona_function",
            ),
            (
                not_ported(store.remove_persona_function(id, "f").await),
                "PersonaStore::remove_persona_function",
            ),
            (
                not_ported(store.add_persona_decision(id, x, 0.5).await),
                "PersonaStore::add_persona_decision",
            ),
            (
                not_ported(store.remove_persona_decision(id, x).await),
                "PersonaStore::remove_persona_decision",
            ),
            (
                not_ported(store.find_personas_for_file("src/a.rs", p.id).await),
                "PersonaStore::find_personas_for_file",
            ),
            (
                not_ported(store.get_all_persona_knows(p.id).await),
                "PersonaStore::get_all_persona_knows",
            ),
            (
                not_ported(store.maintain_personas(p.id).await),
                "PersonaStore::maintain_personas",
            ),
            (
                not_ported(store.detect_personas(p.id).await),
                "PersonaStore::detect_personas",
            ),
            (
                not_ported(store.find_adjacent_personas("src/a.rs", p.id).await),
                "PersonaStore::find_adjacent_personas",
            ),
            (
                not_ported(
                    store
                        .find_relevant_personas_for_note(&["src/a.rs".to_string()], p.id)
                        .await,
                ),
                "PersonaStore::find_relevant_personas_for_note",
            ),
            (
                not_ported(store.find_relevant_personas_for_decision(x, p.id).await),
                "PersonaStore::find_relevant_personas_for_decision",
            ),
            (
                not_ported(store.auto_link_decision_to_persona(id, x, 0.5).await),
                "PersonaStore::auto_link_decision_to_persona",
            ),
            (
                not_ported(store.auto_link_file_to_persona(id, "src/a.rs", 0.5).await),
                "PersonaStore::auto_link_file_to_persona",
            ),
            (
                not_ported(store.compute_persona_affinity(id, x).await),
                "PersonaStore::compute_persona_affinity",
            ),
            (
                not_ported(store.merge_personas(id, x).await),
                "PersonaStore::merge_personas",
            ),
            (
                not_ported(store.find_synapse_linked_personas(id).await),
                "PersonaStore::find_synapse_linked_personas",
            ),
            (
                not_ported(
                    store
                        .add_entity_to_feature_graph(fg.id, "file", "src/a.rs", None, Some(p.id))
                        .await,
                ),
                "FeatureGraphStore::add_entity_to_feature_graph",
            ),
            (
                not_ported(
                    store
                        .remove_entity_from_feature_graph(fg.id, "file", "src/a.rs")
                        .await,
                ),
                "FeatureGraphStore::remove_entity_from_feature_graph",
            ),
            (
                not_ported(
                    store
                        .auto_build_feature_graph("x", None, p.id, "main", 2, None, None)
                        .await,
                ),
                "FeatureGraphStore::auto_build_feature_graph",
            ),
            (
                not_ported(store.refresh_feature_graph(fg.id).await),
                "FeatureGraphStore::refresh_feature_graph",
            ),
            (
                not_ported(store.get_top_entry_functions(p.id, 5).await),
                "FeatureGraphStore::get_top_entry_functions",
            ),
            (
                not_ported(store.get_feature_graph_statistics(fg.id).await),
                "FeatureGraphStore::get_feature_graph_statistics",
            ),
            (
                not_ported(store.compare_feature_graphs(fg.id, x).await),
                "FeatureGraphStore::compare_feature_graphs",
            ),
            (
                not_ported(store.find_overlapping_feature_graphs(fg.id, 0.5).await),
                "FeatureGraphStore::find_overlapping_feature_graphs",
            ),
        ];
        let names: Vec<&str> = cas.iter().map(|(_, n)| *n).collect();
        for (got, name) in cas {
            assert_eq!(got, NonPorte(name));
        }
        // The test covers each listed refusal, and only them.
        let listed: Vec<&str> = refusees_persona()
            .iter()
            .chain(refusees_featuregraph())
            .copied()
            .collect();
        assert_eq!(names, listed);
        // Written by hand (the build checks it), not counted as serving.
        let written = methodes_portees();
        let servent = methodes_qui_servent();
        assert!(listed
            .iter()
            .all(|m| written.contains(m) && !servent.contains(m)));
        let persona_served = servent
            .iter()
            .filter(|m| m.starts_with("PersonaStore::"))
            .count();
        let graph_served = servent
            .iter()
            .filter(|m| m.starts_with("FeatureGraphStore::"))
            .count();
        assert_eq!((persona_served, graph_served), (15, 5));
        // Nothing was written by the refusals: after a reopen (the unresolved edges are
        // only seen there), nothing is quarantined and nothing refuses that served before.
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert_eq!(rapport, RapportOuverture::default(), "{rapport:?}");
        let sg = store.get_persona_subgraph(id).await.unwrap();
        assert_eq!(sg.stats.total_entities, 0);
        let detail = store
            .get_feature_graph_detail(fg.id)
            .await
            .unwrap()
            .unwrap();
        assert!(detail.entities.is_empty());
        assert_eq!(
            store.list_feature_graphs(Some(p.id)).await.unwrap()[0].entity_count,
            Some(0)
        );
    }

    /// A node of a foreign base, by label and properties (no `created_at`: unreadable).
    fn unreadable_persona(db: &obrain_engine::ObrainDB, id: Uuid) -> obrain_common::types::NodeId {
        db.create_node_with_props(
            &["Persona"],
            vec![
                ("id", Value::from(id.to_string())),
                ("name", Value::from("illisible")),
                ("status", Value::from("active")),
                ("energy", Value::from(0.5)),
            ],
        )
    }

    fn node_of(db: &obrain_engine::ObrainDB, id: Uuid) -> obrain_common::types::NodeId {
        db.find_nodes_by_property("id", &Value::from(id.to_string()))[0]
    }

    /// A EXTENDS B EXTENDS C, C unreadable (quarantined): A's ancestors would read [B]
    /// where Neo4j reads [B, C]. The subgraph of A and of B refuses by name.
    #[tokio::test]
    async fn a_subgraph_whose_ancestor_extends_an_unreadable_persona_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let a = a_persona(None, "a", 0.5);
        let b = a_persona(None, "b", 0.5);
        store.create_persona(&a).await.unwrap();
        store.create_persona(&b).await.unwrap();
        store.add_persona_extends(a.id, b.id).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let (bid, c) = (b.id, Uuid::new_v4());
        write_foreign(&base, move |db| {
            let cn = unreadable_persona(db, c);
            db.create_edge(node_of(db, bid), cn, "EXTENDS");
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        for id in [a.id, b.id] {
            assert_eq!(
                not_ported(store.get_persona_subgraph(id).await),
                NonPorte("PersonaStore::get_persona_subgraph"),
                "{id}"
            );
        }
    }

    /// D, unreadable (quarantined), EXTENDS A: A's children would omit D. Refused by name.
    #[tokio::test]
    async fn a_subgraph_with_an_unreadable_child_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let a = a_persona(None, "a", 0.5);
        store.create_persona(&a).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let (aid, d) = (a.id, Uuid::new_v4());
        write_foreign(&base, move |db| {
            let dn = unreadable_persona(db, d);
            db.create_edge(dn, node_of(db, aid), "EXTENDS");
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 1);
        assert_eq!(
            not_ported(store.get_persona_subgraph(a.id).await),
            NonPorte("PersonaStore::get_persona_subgraph")
        );
    }

    /// A persona deletion, and a feature-graph deletion, interrupted after each of their
    /// engine writes: the reopened base has finished the deletion. Never a node served
    /// as whole with part of its edges gone.
    #[tokio::test]
    async fn an_interrupted_persona_or_feature_graph_deletion_is_finished_at_open() {
        use project_orchestrator::notes::{Note, NoteType};
        let dir = tempfile::tempdir().unwrap();
        let mut atteints = 0;
        for graphe in [false, true] {
            for k in 1..=8 {
                let base = dir.path().join(format!("del-{graphe}-{k}"));
                let store = open(&base).await;
                let p = a_project("deletion");
                store.create_project(&p).await.unwrap();
                let fg = a_feature_graph(p.id, "fg");
                store.create_feature_graph(&fg).await.unwrap();
                let parent = a_persona(None, "parent", 0.5);
                let child = a_persona(Some(p.id), "child", 0.5);
                store.create_persona(&parent).await.unwrap();
                store.create_persona(&child).await.unwrap();
                let n = Note::new(Some(p.id), NoteType::Tip, "n".into(), "alice".into());
                store.create_note(&n).await.unwrap();
                store.add_persona_note(child.id, n.id, 0.5).await.unwrap();
                store
                    .add_persona_extends(child.id, parent.id)
                    .await
                    .unwrap();
                store
                    .set_persona_feature_graph(child.id, fg.id)
                    .await
                    .unwrap();
                store.interrompre_au_prochain(Interruption::AvantEtape(k));
                let issue = if graphe {
                    store.delete_feature_graph(fg.id).await.map(|_| ())
                } else {
                    store.delete_persona(child.id).await.map(|_| ())
                };
                if issue.is_ok() {
                    store.close().await.unwrap();
                    continue;
                }
                atteints += 1;
                store.close().await.unwrap();
                drop(store);
                let store = open(&base).await;
                let rapport = store.rapport_d_ouverture().clone();
                assert!(rapport.en_quarantaine.is_empty(), "{k}: {rapport:?}");
                assert_eq!(
                    rapport.suppressions_achevees, 1,
                    "{graphe} {k}: {rapport:?}"
                );
                if graphe {
                    assert!(store.get_feature_graph(fg.id).await.unwrap().is_none());
                    let sg = store.get_persona_subgraph(child.id).await.unwrap();
                    assert_eq!(sg.feature_graph_id, None);
                    assert_eq!(sg.notes.len(), 1);
                } else {
                    assert!(store.get_persona(child.id).await.unwrap().is_none());
                    assert!(store
                        .get_persona_subgraph(parent.id)
                        .await
                        .unwrap()
                        .children
                        .is_empty());
                }
                store.close().await.unwrap();
            }
        }
        // Persona: USES, EXTENDS, SCOPED_TO, BELONGS_TO, then the node; graph: BELONGS_TO,
        // SCOPED_TO, then the node.
        assert!(atteints >= 6, "the crash points were reached: {atteints}");
    }

    /// Removing a link to a note or a persona that is on disk but quarantined removes the
    /// edge (Neo4j's MATCH does not depend on the target being readable), and the
    /// subgraph serves again after a reopen.
    #[tokio::test]
    async fn removing_a_link_to_a_quarantined_target_removes_the_edge() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let a = a_persona(None, "a", 0.5);
        store.create_persona(&a).await.unwrap();
        store.close().await.unwrap();
        drop(store);
        let (aid, note, parent) = (a.id, Uuid::new_v4(), Uuid::new_v4());
        write_foreign(&base, move |db| {
            // A note with no `created_at`, and a persona with none either.
            let nn = db.create_node_with_props(
                &["Note"],
                vec![
                    ("id", Value::from(note.to_string())),
                    ("status", Value::from("active")),
                    ("note_type", Value::from("tip")),
                    ("content", Value::from("illisible")),
                    ("created_by", Value::from("alice")),
                ],
            );
            let pn = unreadable_persona(db, parent);
            let an = node_of(db, aid);
            db.create_edge_with_props(an, nn, "USES", [("weight", Value::from(0.5))]);
            db.create_edge(an, pn, "EXTENDS");
        })
        .await;
        let store = open(&base).await;
        assert_eq!(store.rapport_d_ouverture().en_quarantaine.len(), 2);
        assert!(store.get_persona_subgraph(a.id).await.is_err());
        store.remove_persona_note(a.id, note).await.unwrap();
        store.remove_persona_extends(a.id, parent).await.unwrap();
        assert!(store.get_persona_subgraph(a.id).await.is_ok());
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        let sg = store.get_persona_subgraph(a.id).await.unwrap();
        assert!(sg.notes.is_empty() && sg.parents.is_empty());
    }

    /// A migrated persona whose `status` or `energy` is absent, or whose `project_id` is
    /// not a UUID, is quarantined: Neo4j would decode it with defaults and filter / sort /
    /// list it on the stored value, so this backend would answer differently.
    #[tokio::test]
    async fn an_incomplete_migrated_persona_is_quarantined() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        store.close().await.unwrap();
        drop(store);
        let ids = [Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        write_foreign(&base, move |db| {
            let now = Value::from(Utc::now().to_rfc3339());
            let base_props = |id: Uuid| {
                vec![
                    ("id", Value::from(id.to_string())),
                    ("name", Value::from("migrée")),
                    ("created_at", now.clone()),
                ]
            };
            let mut sans_statut = base_props(ids[0]);
            sans_statut.push(("energy", Value::from(0.5)));
            db.create_node_with_props(&["Persona"], sans_statut);
            let mut sans_energie = base_props(ids[1]);
            sans_energie.push(("status", Value::from("active")));
            db.create_node_with_props(&["Persona"], sans_energie);
            let mut projet_illisible = base_props(ids[2]);
            projet_illisible.extend([
                ("status", Value::from("active")),
                ("energy", Value::from(0.5)),
                ("project_id", Value::from("pas-un-uuid")),
            ]);
            db.create_node_with_props(&["Persona"], projet_illisible);
        })
        .await;
        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert_eq!(rapport.en_quarantaine.len(), 3, "{rapport:?}");
        for id in ids {
            assert!(store.get_persona(id).await.unwrap().is_none());
        }
        assert!(store.list_global_personas().await.unwrap().is_empty());
    }

    /// Every persona and feature-graph write that carries a value past the blob threshold
    /// (a description, a system prompt, a JSON string) — or that replaces or removes one —
    /// pays the full flush, before the commit point for a creation. Observed through the
    /// engine's own fsync counter: no `close()` is involved, which would flush anyway.
    #[tokio::test]
    async fn persona_and_feature_graph_writes_past_the_blob_threshold_pay_the_full_flush() {
        let store = temporary(Durabilite::Wal).await;
        let p = a_project("blob");
        store.create_project(&p).await.unwrap();
        let long = "x".repeat(257);
        // Every case is checked; the failures are reported together.
        let mut manquants: Vec<String> = Vec::new();
        macro_rules! cout {
            ($e:expr) => {{
                let t0 = store.fsyncs_du_wal().unwrap();
                $e.await.unwrap();
                store.fsyncs_du_wal().unwrap() - t0
            }};
        }
        // Creations.
        let court = cout!(store.create_persona(&a_persona(Some(p.id), "court", 0.5)));
        assert!(court >= 1, "the WAL fsync ran");
        let mut desc = a_persona(Some(p.id), "desc", 0.5);
        desc.description = long.clone();
        let mut prompt = a_persona(Some(p.id), "prompt", 0.5);
        prompt.system_prompt_override = Some(long.clone());
        let mut histoire = a_persona(Some(p.id), "histoire", 0.5);
        histoire.energy_history = vec![0.123_456_789_012_345; 30];
        for (cas, x) in [
            ("description", &desc),
            ("system prompt", &prompt),
            ("JSON", &histoire),
        ] {
            let c = cout!(store.create_persona(x));
            if c <= court {
                manquants.push(format!(
                    "create_persona with a long {cas}: {c} fsyncs against {court}"
                ));
            }
        }
        // Updates: a new long value, an old long value replaced, an old long value removed.
        let mut u = a_persona(Some(p.id), "maj", 0.5);
        store.create_persona(&u).await.unwrap();
        let court_maj = cout!(store.update_persona(&u));
        u.model_preference = Some(long.clone());
        let c = cout!(store.update_persona(&u));
        if c <= court_maj {
            manquants.push(format!("new long value: {c} against {court_maj}"));
        }
        let mut d = desc.clone();
        d.description = "court".into();
        let c = cout!(store.update_persona(&d));
        if c <= court_maj {
            manquants.push(format!(
                "old long description replaced: {c} against {court_maj}"
            ));
        }
        let mut q = prompt.clone();
        q.system_prompt_override = None;
        let c = cout!(store.update_persona(&q));
        if c <= court_maj {
            manquants.push(format!("old long prompt removed: {c} against {court_maj}"));
        }
        // Feature graphs.
        let court_fg = cout!(store.create_feature_graph(&a_feature_graph(p.id, "court")));
        let mut g = a_feature_graph(p.id, "desc");
        g.description = Some(long.clone());
        let c = cout!(store.create_feature_graph(&g));
        if c <= court_fg {
            manquants.push(format!("long description: {c} against {court_fg}"));
        }
        let mut g = a_feature_graph(p.id, "rel");
        g.include_relations = Some(vec!["IMPLEMENTS_TRAIT".to_string(); 20]);
        let c = cout!(store.create_feature_graph(&g));
        if c <= court_fg {
            manquants.push(format!("long JSON: {c} against {court_fg}"));
        }
        assert!(manquants.is_empty(), "no full flush for: {manquants:#?}");
        // The removal is applied.
        let relu = store.get_persona(prompt.id).await.unwrap().unwrap();
        assert_eq!(relu.system_prompt_override, None);
    }

    // ------------------------------------------------------------------
    // Skills et partage
    // ------------------------------------------------------------------

    fn un_skill_publie(source: Uuid) -> project_orchestrator::skills::registry::PublishedSkill {
        serde_json::from_value(serde_json::json!({
            "id": Uuid::new_v4(),
            "name": "Publié",
            "description": "un skill du registre",
            "tags": ["a", "b"],
            "package": {
                "schema_version": 3,
                "metadata": {"format": "po-skill/v3", "exported_at": "2026-01-01T00:00:00Z",
                    "source_project": "p",
                    "stats": {"note_count": 0, "decision_count": 0, "trigger_count": 0,
                        "activation_count": 0}},
                "skill": {"name": "Publié", "description": "", "trigger_patterns": [],
                    "context_template": null, "tags": [], "cohesion": 0.5},
                "notes": [], "decisions": []
            },
            "trust_score": 0.5,
            "trust_components": {"energy": 0.1, "cohesion": 0.2, "activation": 0.3,
                "success_rate": 0.4, "source_projects": 0.5},
            "trust_level": "medium",
            "source_project_id": source,
            "source_project_name": "p",
            "published_by": "agent",
            "published_at": "2026-01-01T00:00:00Z",
            "import_count": 0
        }))
        .unwrap()
    }

    /// Skills (projet et transverse), membre note, activation, registre : écrits,
    /// relus, puis relus à l'identique APRÈS fermeture et réouverture ; ensuite
    /// rescope, retrait, suppression de la note membre (DETACH) et du skill.
    #[tokio::test]
    async fn skills_et_registre_survivent_une_reouverture() {
        use project_orchestrator::notes::{Note, NoteType};
        use project_orchestrator::skills::{SkillNode, SkillStatus, SkillTrigger};
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("skills-reopen");
        store.create_project(&p).await.unwrap();
        let note = Note::new(
            Some(p.id),
            NoteType::Tip,
            "membre".to_string(),
            "alice".to_string(),
        );
        store.create_note(&note).await.unwrap();

        let mut s = SkillNode::new(Some(p.id), "neo4j");
        s.status = SkillStatus::Active;
        s.energy = 0.9;
        s.trigger_patterns = vec![SkillTrigger::regex("neo4j|cypher", 0.7)];
        s.tags = vec!["db".into()];
        s.context_template = Some("tpl".into());
        store.create_skill(&s).await.unwrap();
        let mut t = SkillNode::new(None, "transverse");
        t.status = SkillStatus::Active;
        t.energy = 0.1;
        store.create_skill(&t).await.unwrap();

        store.add_skill_member(s.id, "note", note.id).await.unwrap();
        // MERGE: a second add draws no second edge.
        store.add_skill_member(s.id, "note", note.id).await.unwrap();
        store.increment_skill_activation(s.id).await.unwrap();
        let publiee = un_skill_publie(p.id);
        store.upsert_published_skill(&publiee).await.unwrap();
        store
            .increment_published_skill_imports(publiee.id)
            .await
            .unwrap();

        let avant = store.get_skill(s.id).await.unwrap().unwrap();
        assert_eq!(avant.note_count, 1);
        assert_eq!(avant.activation_count, 1);
        assert!(avant.last_activated.is_some());

        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert!(rapport.en_quarantaine.is_empty(), "{rapport:?}");
        let apres = store.get_skill(s.id).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&apres).unwrap(),
            serde_json::to_value(&avant).unwrap()
        );
        let (membres, decisions) = store.get_skill_members(s.id).await.unwrap();
        assert_eq!(membres.iter().map(|n| n.id).collect::<Vec<_>>(), [note.id]);
        assert!(decisions.is_empty());
        let ids = |v: Vec<SkillNode>| v.into_iter().map(|k| k.id).collect::<Vec<_>>();
        assert_eq!(
            ids(store.get_skills_for_note(note.id).await.unwrap()),
            [s.id]
        );
        // The scope rule: a project sees its skills AND the transverse ones.
        let (liste, total) = store.list_skills(Some(p.id), None, 10, 0).await.unwrap();
        assert_eq!((ids(liste), total), (vec![s.id, t.id], 2));
        assert_eq!(
            ids(store.list_skills(None, None, 10, 0).await.unwrap().0),
            [t.id]
        );
        assert_eq!(
            ids(store.get_skills_for_project(p.id).await.unwrap()),
            [s.id]
        );
        assert_eq!(
            ids(store.get_live_skills_for_project(p.id).await.unwrap()),
            [s.id]
        );
        let declenches = store
            .match_skills_by_trigger(Some(p.id), "une requête CYPHER")
            .await
            .unwrap();
        assert_eq!(
            declenches
                .iter()
                .map(|(k, c)| (k.id, *c))
                .collect::<Vec<_>>(),
            [(s.id, 1.0)]
        );
        let ctx = store.activate_skill(s.id, "q").await.unwrap();
        assert_eq!(ctx.activated_notes.len(), 1);
        assert!(ctx.context_text.starts_with("## Skill: neo4j"));
        assert_eq!(
            store
                .get_skill(s.id)
                .await
                .unwrap()
                .unwrap()
                .activation_count,
            2
        );
        let relu = store
            .get_published_skill(publiee.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(relu.import_count, 1);
        assert_eq!(relu.tags, publiee.tags);
        let (trouves, total) = store
            .search_published_skills(Some("PUBL"), Some(0.1), None, 10, 0)
            .await
            .unwrap();
        assert_eq!((trouves.len(), total), (1, 1));
        assert_eq!(
            store
                .search_published_skills(None, Some(0.9), None, 10, 0)
                .await
                .unwrap()
                .1,
            0
        );

        store.rescope_skill(s.id, None).await.unwrap();
        assert_eq!(
            store.get_skill(s.id).await.unwrap().unwrap().project_id,
            None
        );
        assert!(store
            .remove_skill_member(s.id, "note", note.id)
            .await
            .unwrap());
        assert!(!store
            .remove_skill_member(s.id, "note", note.id)
            .await
            .unwrap());
        assert_eq!(store.get_skill(s.id).await.unwrap().unwrap().note_count, 0);
        store.add_skill_member(s.id, "note", note.id).await.unwrap();
        // Deleting the note detaches its MEMBER_OF: the skill no longer serves it.
        assert!(store.delete_note(note.id).await.unwrap());
        assert!(store.get_skill_members(s.id).await.unwrap().0.is_empty());
        assert!(store.get_skills_for_note(note.id).await.unwrap().is_empty());
        assert!(store.delete_skill(s.id).await.unwrap());
        assert!(!store.delete_skill(s.id).await.unwrap());
        store.close().await.unwrap();
        drop(store);

        let store = open(&base).await;
        assert!(store.get_skill(s.id).await.unwrap().is_none());
        assert_eq!(
            store.get_skill(t.id).await.unwrap().unwrap().name,
            "transverse"
        );
        assert_eq!(
            store
                .get_published_skill(publiee.id)
                .await
                .unwrap()
                .unwrap()
                .import_count,
            1
        );
        store.close().await.unwrap();
    }

    /// Une création de skill interrompue à chacune de ses écritures : rien n'est
    /// servi, la réouverture supprime le brouillon (et rien d'autre), et la même
    /// création réussit ensuite.
    #[tokio::test]
    async fn une_creation_de_skill_interrompue_ne_laisse_qu_un_brouillon() {
        use project_orchestrator::skills::SkillNode;
        let dir = tempfile::tempdir().unwrap();
        let mut touches = 0;
        for k in 1..=60 {
            let base = dir.path().join(format!("skill-{k}"));
            let store = open(&base).await;
            let p = a_project("skill-interrompu");
            store.create_project(&p).await.unwrap();
            let s = SkillNode::new(Some(p.id), "interrompu");
            store.interrompre_au_prochain(Interruption::AvantEtape(k));
            if store.create_skill(&s).await.is_ok() {
                // Past the last write of a creation: every point was reached.
                store.close().await.unwrap();
                break;
            }
            touches += 1;
            assert!(store.get_skill(s.id).await.unwrap().is_none(), "{k}");
            store.close().await.unwrap();
            drop(store);

            let store = open(&base).await;
            let rapport = store.rapport_d_ouverture().clone();
            assert_eq!(rapport.brouillons_supprimes, 1, "{k}: {rapport:?}");
            assert!(rapport.en_quarantaine.is_empty(), "{k}: {rapport:?}");
            assert!(store.get_skill(s.id).await.unwrap().is_none(), "{k}");
            assert_eq!(
                store.list_skills(Some(p.id), None, 10, 0).await.unwrap().1,
                0
            );
            store.create_skill(&s).await.unwrap();
            store.close().await.unwrap();
        }
        // The marker, every property, the BELONGS_TO edge, the commit point.
        assert!(touches >= 20, "points atteints : {touches}");
    }

    /// Une base écrite ailleurs : un skill illisible est mis en quarantaine (son id
    /// est réservé), un skill lisible avec un membre Decision est servi mais ses
    /// membres sont refusés par leur nom, un événement de partage sans date est en
    /// quarantaine.
    #[tokio::test]
    async fn une_base_etrangere_met_en_quarantaine_ce_qu_elle_ne_lit_pas() {
        use project_orchestrator::episodes::distill_models::{
            PrivacyMode, SharingConsent, SharingEvent,
        };
        use project_orchestrator::skills::SkillNode;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        store.close().await.unwrap();
        drop(store);
        let illisible = Uuid::new_v4();
        let lisible = Uuid::new_v4();
        let maintenant = Utc::now().to_rfc3339();
        write_foreign(&base, move |db| {
            db.create_node_with_props(
                &["Skill"],
                vec![
                    ("id", Value::from(illisible.to_string())),
                    ("name", Value::from("illisible")),
                    ("status", Value::from("zombie")),
                    ("created_at", Value::from(maintenant.clone())),
                    ("updated_at", Value::from(maintenant.clone())),
                ],
            );
            let s = db.create_node_with_props(
                &["Skill"],
                vec![
                    ("id", Value::from(lisible.to_string())),
                    ("name", Value::from("lisible")),
                    ("status", Value::from("active")),
                    ("created_at", Value::from(maintenant.clone())),
                    ("updated_at", Value::from(maintenant)),
                ],
            );
            let d = db.create_node_with_props(
                &["Decision"],
                vec![("id", Value::from(Uuid::new_v4().to_string()))],
            );
            db.create_edge(d, s, "MEMBER_OF_SKILL");
            db.create_node_with_props(&["SharingEvent"], vec![("id", Value::from("ev-sans-date"))]);
        })
        .await;

        let store = open(&base).await;
        let rapport = store.rapport_d_ouverture().clone();
        assert!(
            rapport.en_quarantaine.iter().any(|q| q.contains("zombie")),
            "{rapport:?}"
        );
        assert!(
            rapport
                .en_quarantaine
                .iter()
                .any(|q| q.contains("SharingEvent") && q.contains("timestamp")),
            "{rapport:?}"
        );
        assert_eq!(rapport.brouillons_supprimes, 0);
        // The unreadable skill: not served, its id not reusable, not deleted blindly.
        assert!(store.get_skill(illisible).await.unwrap().is_none());
        let mut s = SkillNode::new(None, "nouveau");
        s.id = illisible;
        let err = store.create_skill(&s).await.unwrap_err();
        assert!(format!("{err:#}").contains("réservé"), "{err:#}");
        assert!(store.delete_skill(illisible).await.is_err());
        // The readable one is served; its members are refused by name.
        assert_eq!(
            store.get_skill(lisible).await.unwrap().unwrap().name,
            "lisible"
        );
        assert_eq!(
            store
                .get_skill_members(lisible)
                .await
                .unwrap_err()
                .downcast_ref::<NonPorte>(),
            Some(&NonPorte("SkillStore::get_skill_members"))
        );
        assert_eq!(
            store
                .activate_skill(lisible, "q")
                .await
                .unwrap_err()
                .downcast_ref::<NonPorte>(),
            Some(&NonPorte("SkillStore::activate_skill"))
        );
        // The event: its id is reserved.
        let ev = SharingEvent {
            id: "ev-sans-date".into(),
            content_hash: "h".into(),
            artifact_type: "note".into(),
            action: "shared".into(),
            source_did: "did:a".into(),
            target_did: "broadcast".into(),
            timestamp: Utc::now(),
            consent: SharingConsent::NotSet,
            privacy_mode: PrivacyMode::Standard,
            reason: None,
        };
        let err = store.create_sharing_event(&ev).await.unwrap_err();
        assert!(format!("{err:#}").contains("réservé"), "{err:#}");
        // Removing every member deletes the Decision edge without the Decision domain;
        // the members are then served.
        assert_eq!(store.remove_all_skill_members(lisible).await.unwrap(), 1);
        let (notes, decisions) = store.get_skill_members(lisible).await.unwrap();
        assert!(notes.is_empty() && decisions.is_empty());
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        assert!(store.get_skill_members(lisible).await.is_ok());
        store.close().await.unwrap();
    }

    /// Chaque méthode des registres REFUSEES de skill.rs et sharing.rs refuse par
    /// son nom ; les refus conditionnels aussi, et rien n'est écrit.
    #[tokio::test]
    async fn les_refus_skill_et_partage_sont_nommes_et_n_ecrivent_rien() {
        use project_orchestrator::episodes::distill_models::SharingConsent;
        use project_orchestrator::skills::SkillNode;
        let store = temporary(Durabilite::Wal).await;
        let id = Uuid::new_v4();
        let mut verifiees = 0;
        for nom in refusees_skill().iter().chain(refusees_sharing()) {
            let resultat = match *nom {
                "SkillStore::create_used_skill_relation" => {
                    store.create_used_skill_relation(id, id, "success").await
                }
                "SharingStore::update_decision_sharing_consent" => {
                    store
                        .update_decision_sharing_consent(id, &SharingConsent::ExplicitAllow)
                        .await
                }
                "SharingStore::list_sharing_events" => {
                    store.list_sharing_events(id, 10, 0).await.map(|_| ())
                }
                autre => panic!("refus sans vérification : {autre}"),
            };
            assert_eq!(
                resultat.unwrap_err().downcast_ref::<NonPorte>(),
                Some(&NonPorte(nom))
            );
            verifiees += 1;
        }
        assert_eq!(verifiees, 3);

        // Conditional refusals: a Decision member, a tags filter.
        let p = a_project("skill-refus");
        store.create_project(&p).await.unwrap();
        let s = SkillNode::new(Some(p.id), "refus");
        store.create_skill(&s).await.unwrap();
        let avant = serde_json::to_value(store.get_skill(s.id).await.unwrap()).unwrap();
        assert_eq!(
            not_ported(store.add_skill_member(s.id, "decision", id).await),
            NonPorte("SkillStore::add_skill_member")
        );
        assert_eq!(
            not_ported(store.remove_skill_member(s.id, "decision", id).await),
            NonPorte("SkillStore::remove_skill_member")
        );
        let tags = ["a".to_string()];
        assert_eq!(
            not_ported(
                store
                    .search_published_skills(None, None, Some(&tags), 10, 0)
                    .await
            ),
            NonPorte("SkillStore::search_published_skills")
        );
        assert_eq!(
            serde_json::to_value(store.get_skill(s.id).await.unwrap()).unwrap(),
            avant
        );
        // Plain errors, before any write: an unknown member type, an unknown
        // project, a published skill without source project.
        let err = store.add_skill_member(s.id, "plan", id).await.unwrap_err();
        assert!(err.downcast_ref::<NonPorte>().is_none());
        let orphelin = SkillNode::new(Some(Uuid::new_v4()), "orphelin");
        assert!(store.create_skill(&orphelin).await.is_err());
        assert!(store.get_skill(orphelin.id).await.unwrap().is_none());
        let mut sans_source = un_skill_publie(p.id);
        sans_source.source_project_id = None;
        assert!(store.upsert_published_skill(&sans_source).await.is_err());
        assert!(store
            .get_published_skill(sans_source.id)
            .await
            .unwrap()
            .is_none());
    }

    /// Politique, consentement, événements et tombes : écrits, relus, puis relus
    /// à l'identique après fermeture et réouverture.
    #[tokio::test]
    async fn le_partage_survit_une_reouverture() {
        use project_orchestrator::episodes::distill_models::{
            ConsentCounts, PrivacyMode, SharingConsent, SharingEvent, SharingPolicy,
        };
        use project_orchestrator::notes::{Note, NoteType};
        use project_orchestrator::reception::anchor::SignedTombstone;
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("partage");
        store.create_project(&p).await.unwrap();
        assert!(store.get_sharing_policy(p.id).await.unwrap().is_none());
        let politique = SharingPolicy {
            min_shareability_score: 0.9,
            ..Default::default()
        };
        store.update_sharing_policy(p.id, &politique).await.unwrap();
        // An unknown project: Neo4j's MATCH, nothing written.
        store
            .update_sharing_policy(Uuid::new_v4(), &politique)
            .await
            .unwrap();

        let n1 = Note::new(Some(p.id), NoteType::Tip, "a".into(), "alice".into());
        let n2 = Note::new(None, NoteType::Tip, "b".into(), "alice".into());
        store.create_note(&n1).await.unwrap();
        store.create_note(&n2).await.unwrap();
        store
            .update_sharing_consent(n1.id, &SharingConsent::ExplicitAllow)
            .await
            .unwrap();

        let ev = |id: &str, secondes: i64| SharingEvent {
            id: id.into(),
            content_hash: format!("h-{id}"),
            artifact_type: "note".into(),
            action: "shared".into(),
            source_did: "did:a".into(),
            target_did: "broadcast".into(),
            timestamp: Utc::now() + chrono::Duration::seconds(secondes),
            consent: SharingConsent::PolicyAuto,
            privacy_mode: PrivacyMode::Strict,
            reason: Some("audit".into()),
        };
        let (ancien, recent) = (ev("ev-1", -60), ev("ev-2", 0));
        store.create_sharing_event(&ancien).await.unwrap();
        store.create_sharing_event(&recent).await.unwrap();
        assert!(store.create_sharing_event(&recent).await.is_err());

        let identite = project_orchestrator::identity::InstanceIdentity::generate();
        let vraie = project_orchestrator::sharing::tombstone::sign_tombstone(
            &identite,
            "hash-vrai".into(),
            Utc::now(),
            Some("rgpd".into()),
        );
        let factice = SignedTombstone {
            content_hash: "hash-factice".into(),
            issuer_did: "did:local:unknown".into(),
            signature_hex: "00".into(),
            issued_at: Utc::now(),
            reason: None,
        };
        store.persist_tombstone(&vraie).await.unwrap();
        store.persist_tombstone(&factice).await.unwrap();
        // MERGE on the hash: a second write of the same hash replaces, never doubles.
        store.persist_tombstone(&factice).await.unwrap();

        let compte = |not_set, explicit_allow| ConsentCounts {
            not_set,
            explicit_allow,
            explicit_deny: 0,
            policy_auto: 0,
        };
        {
            let passe = "avant la réouverture";
            let relue = store.get_sharing_policy(p.id).await.unwrap().unwrap();
            assert_eq!(
                serde_json::to_value(&relue).unwrap(),
                serde_json::to_value(&politique).unwrap(),
                "{passe}"
            );
            // The project index sees the policy too.
            assert!(
                store
                    .get_project(p.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .sharing_policy
                    .is_some(),
                "{passe}"
            );
            assert_eq!(
                store.get_sharing_consent(n1.id).await.unwrap(),
                SharingConsent::ExplicitAllow,
                "{passe}"
            );
            assert_eq!(
                store
                    .get_note(n1.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .sharing_consent,
                SharingConsent::ExplicitAllow,
                "{passe}"
            );
            assert_eq!(
                store.get_sharing_consent(n2.id).await.unwrap(),
                SharingConsent::NotSet
            );
            assert_eq!(
                store.get_sharing_consent(Uuid::new_v4()).await.unwrap(),
                SharingConsent::NotSet
            );
            assert_eq!(
                store.count_notes_by_consent(Some(p.id)).await.unwrap(),
                compte(0, 1),
                "{passe}"
            );
            assert_eq!(
                store.count_notes_by_consent(None).await.unwrap(),
                compte(1, 1),
                "{passe}"
            );
            let tombes = store.list_tombstones().await.unwrap();
            assert_eq!(tombes.len(), 2, "{passe}");
            let relue = tombes
                .iter()
                .find(|t| t.content_hash == "hash-vrai")
                .unwrap();
            assert!(project_orchestrator::sharing::tombstone::verify_tombstone(
                relue
            ));
            assert!(store.is_tombstoned("hash-vrai").await.unwrap(), "{passe}");
            assert!(!store.is_tombstoned("hash-factice").await.unwrap());
            assert!(!store.is_tombstoned("inconnu").await.unwrap());
        }
        store.close().await.unwrap();
        drop(store);
        let store = open(&base).await;
        assert!(store.rapport_d_ouverture().en_quarantaine.is_empty());
        let relue = store.get_sharing_policy(p.id).await.unwrap().unwrap();
        assert_eq!(
            serde_json::to_value(&relue).unwrap(),
            serde_json::to_value(&politique).unwrap()
        );
        assert_eq!(
            store.get_sharing_consent(n1.id).await.unwrap(),
            SharingConsent::ExplicitAllow
        );
        assert_eq!(
            store.count_notes_by_consent(None).await.unwrap(),
            compte(1, 1)
        );
        // The events were durable: their ids are still taken after the reopen.
        for ev in [&ancien, &recent] {
            let err = store.create_sharing_event(ev).await.unwrap_err();
            assert!(format!("{err:#}").contains("existe déjà"), "{err:#}");
        }
        assert_eq!(store.list_tombstones().await.unwrap().len(), 2);
        assert!(store.is_tombstoned("hash-vrai").await.unwrap());
        assert!(!store.is_tombstoned("hash-factice").await.unwrap());
        assert_eq!(
            not_ported(
                store
                    .update_decision_sharing_consent(Uuid::new_v4(), &SharingConsent::NotSet)
                    .await
            ),
            NonPorte("SharingStore::update_decision_sharing_consent")
        );
        store.close().await.unwrap();
    }
    /// Relecture de #1213 (C1, C2, C3, M4) : une base importée, écrite hors de PO,
    /// qui ne doit jamais être lue plus permissive que la référence.
    struct Importee {
        _dir: tempfile::TempDir,
        store: ObrainGraphStore,
        projet: Uuid,
        refus: Uuid,
        vide: Uuid,
        actif: Uuid,
        ok: Uuid,
    }

    async fn base_importee() -> Importee {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let store = open(&base).await;
        let p = a_project("import-strict");
        store.create_project(&p).await.unwrap();
        store.close().await.unwrap();
        drop(store);

        let (refus, vide, actif) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let (ok, corrompue, sans_energie) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
        let pid = p.id.to_string();
        write_foreign(&base, move |db| {
            let t = Utc::now().to_rfc3339();
            let note =
                |id: Uuid, statut: &str, energie: Option<f64>, consentement: Option<&str>| {
                    let mut props = vec![
                        ("id", Value::from(id.to_string())),
                        ("status", Value::from(statut)),
                        ("note_type", Value::from("tip")),
                        ("created_by", Value::from("alice")),
                        ("content", Value::from(format!("note {id}"))),
                        ("created_at", Value::from(t.clone())),
                        ("project_id", Value::from(pid.clone())),
                    ];
                    if let Some(e) = energie {
                        props.push(("energy", Value::from(e)));
                    }
                    if let Some(c) = consentement {
                        props.push(("sharing_consent", Value::from(c)));
                    }
                    db.create_node_with_props(&["Note"], props)
                };
            // C1: a refusal stored in a form the decoder does not know.
            note(refus, "active", Some(0.5), Some("ExplicitDeny"));
            // C2: the legacy empty scope.
            db.create_node_with_props(
                &["Skill"],
                vec![
                    ("id", Value::from(vide.to_string())),
                    ("name", Value::from("vide")),
                    ("status", Value::from("active")),
                    ("project_id", Value::from("")),
                    ("created_at", Value::from(t.clone())),
                    ("updated_at", Value::from(t.clone())),
                ],
            );
            // C3: a placeholder and the real tombstone, same hash.
            for (emetteur, signature) in [("did:local:unknown", "00"), ("did:key:zreel", "abcd")] {
                db.create_node_with_props(
                    &["Tombstone"],
                    vec![
                        ("content_hash", Value::from("h-double")),
                        ("issuer_did", Value::from(emetteur)),
                        ("signature_hex", Value::from(signature)),
                        ("issued_at", Value::from(t.clone())),
                    ],
                );
            }
            // M4: one current member, one with an unknown status, one without energy.
            let s = db.create_node_with_props(
                &["Skill"],
                vec![
                    ("id", Value::from(actif.to_string())),
                    ("name", Value::from("actif")),
                    ("status", Value::from("active")),
                    ("project_id", Value::from(pid.clone())),
                    ("created_at", Value::from(t.clone())),
                    ("updated_at", Value::from(t.clone())),
                ],
            );
            for n in [
                note(ok, "active", Some(0.5), None),
                note(corrompue, "archivee-corrompue", Some(0.9), None),
                note(sans_energie, "active", None, None),
            ] {
                db.create_edge(n, s, "MEMBER_OF");
            }
        })
        .await;

        let store = open(&base).await;
        Importee {
            _dir: dir,
            store,
            projet: p.id,
            refus,
            vide,
            actif,
            ok,
        }
    }

    /// C1 : un consentement PRÉSENT mais illisible met la note en quarantaine ; il
    /// n'est jamais lu `NotSet` (ouvert au partage).
    #[tokio::test]
    async fn un_consentement_illisible_met_la_note_en_quarantaine() {
        use project_orchestrator::neo4j::traits::NoteStore;
        let b = base_importee().await;
        let rapport = b.store.rapport_d_ouverture().clone();
        assert!(
            rapport
                .en_quarantaine
                .iter()
                .any(|q| q.contains("sharing_consent")),
            "{rapport:?}"
        );
        assert!(b.store.get_sharing_consent(b.refus).await.is_err());
        assert!(b.store.get_note(b.refus).await.unwrap().is_none());
        assert!(b.store.count_notes_by_consent(None).await.is_err());
    }

    /// C2 : un skill à `project_id` vide n'est servi à aucun projet (Neo4j ne le
    /// sert à aucun ; lu transverse, il le serait à tous).
    #[tokio::test]
    async fn un_skill_a_portee_vide_n_est_servi_a_aucun_projet() {
        let b = base_importee().await;
        let visibles = b
            .store
            .list_skills(Some(b.projet), None, 50, 0)
            .await
            .unwrap()
            .0;
        assert!(visibles.iter().all(|k| k.id != b.vide), "{visibles:?}");
        let ailleurs = b
            .store
            .list_skills(Some(Uuid::new_v4()), None, 50, 0)
            .await
            .unwrap()
            .0;
        assert!(ailleurs.iter().all(|k| k.id != b.vide), "{ailleurs:?}");
        assert!(b.store.get_skill(b.vide).await.unwrap().is_none());
    }

    /// C3 : deux tombes d'un même hash (un placeholder et la vraie) : la question
    /// « révoqué ? » répond par une erreur, jamais par `false`.
    #[tokio::test]
    async fn deux_tombes_d_un_meme_hash_ne_repondent_pas_non_revoque() {
        let b = base_importee().await;
        assert!(b.store.is_tombstoned("h-double").await.is_err());
        assert!(b.store.list_tombstones().await.is_err());
    }

    /// M4 : l'activation ne retient que les notes courantes au sens des
    /// propriétés brutes (statut connu, énergie présente), comme Neo4j.
    #[tokio::test]
    async fn l_activation_ne_retient_que_les_notes_courantes_brutes() {
        let b = base_importee().await;
        let ctx = b.store.activate_skill(b.actif, "q").await.unwrap();
        assert_eq!(
            ctx.activated_notes
                .iter()
                .map(|a| a.note.id)
                .collect::<Vec<_>>(),
            [b.ok]
        );
    }

    /// Tant qu'obrain ne journalise pas les valeurs en colonne blob (#1190), une
    /// écriture de skill ou de partage qui porte une valeur > 256 o — ou qui en
    /// REMPLACE une — paie le flush complet avant d'être acquittée. Mesuré par le
    /// compteur de fsync du moteur, sans passer par `close()`.
    async fn store_avec_projet() -> (ObrainGraphStore, ProjectNode) {
        let store = temporary(Durabilite::Wal).await;
        let p = a_project("blob");
        store.create_project(&p).await.unwrap();
        (store, p)
    }

    /// The fsyncs one write cost.
    macro_rules! cout {
        ($store:expr, $e:expr) => {{
            let t0 = $store.fsyncs_du_wal().unwrap();
            $e.await.unwrap();
            $store.fsyncs_du_wal().unwrap() - t0
        }};
    }

    fn un_evenement(
        id: &str,
        raison: String,
    ) -> project_orchestrator::episodes::distill_models::SharingEvent {
        use project_orchestrator::episodes::distill_models::{
            PrivacyMode, SharingConsent, SharingEvent,
        };
        SharingEvent {
            id: id.into(),
            content_hash: "h".into(),
            artifact_type: "note".into(),
            action: "shared".into(),
            source_did: "did:a".into(),
            target_did: "broadcast".into(),
            timestamp: Utc::now(),
            consent: SharingConsent::NotSet,
            privacy_mode: PrivacyMode::Standard,
            reason: Some(raison),
        }
    }

    fn une_tombe(
        hash: &str,
        raison: String,
    ) -> project_orchestrator::reception::anchor::SignedTombstone {
        project_orchestrator::reception::anchor::SignedTombstone {
            content_hash: hash.into(),
            issuer_did: "did:key:z".into(),
            signature_hex: "ab".into(),
            issued_at: Utc::now(),
            reason: Some(raison),
        }
    }

    #[tokio::test]
    async fn create_skill_au_dela_du_seuil_paie_le_flush_complet() {
        use project_orchestrator::skills::SkillNode;
        let (store, p) = store_avec_projet().await;
        let mut petit = SkillNode::new(Some(p.id), "petit");
        petit.description = "x".repeat(256);
        let mut grand = SkillNode::new(Some(p.id), "grand");
        grand.description = "x".repeat(257);
        let c_petit = cout!(store, store.create_skill(&petit));
        let c_grand = cout!(store, store.create_skill(&grand));
        assert!(c_grand > c_petit, "{c_grand} contre {c_petit}");
    }

    #[tokio::test]
    async fn update_skill_qui_remplace_une_valeur_longue_paie_le_flush_complet() {
        use project_orchestrator::skills::SkillNode;
        let (store, p) = store_avec_projet().await;
        let petit = SkillNode::new(Some(p.id), "petit");
        let mut grand = SkillNode::new(Some(p.id), "grand");
        grand.description = "x".repeat(257);
        store.create_skill(&petit).await.unwrap();
        store.create_skill(&grand).await.unwrap();
        // The SAME small value written over a short one, and over a long one.
        let mut a = petit.clone();
        a.description = "y".into();
        let mut b = grand.clone();
        b.description = "y".into();
        let c_petit = cout!(store, store.update_skill(&a));
        let c_grand = cout!(store, store.update_skill(&b));
        assert!(c_grand > c_petit, "{c_grand} contre {c_petit}");
    }

    #[tokio::test]
    async fn create_sharing_event_au_dela_du_seuil_paie_le_flush_complet() {
        let (store, _) = store_avec_projet().await;
        let c_petit = cout!(
            store,
            store.create_sharing_event(&un_evenement("e-petit", "r".repeat(256)))
        );
        let c_grand = cout!(
            store,
            store.create_sharing_event(&un_evenement("e-grand", "r".repeat(257)))
        );
        assert!(c_grand > c_petit, "{c_grand} contre {c_petit}");
    }

    #[tokio::test]
    async fn persist_tombstone_au_dela_du_seuil_et_son_remplacement_paient_le_flush_complet() {
        let (store, _) = store_avec_projet().await;
        let c_petit = cout!(
            store,
            store.persist_tombstone(&une_tombe("t-petit", "r".repeat(256)))
        );
        let c_grand = cout!(
            store,
            store.persist_tombstone(&une_tombe("t-grand", "r".repeat(257)))
        );
        assert!(c_grand > c_petit, "création : {c_grand} contre {c_petit}");
        let c_petit = cout!(
            store,
            store.persist_tombstone(&une_tombe("t-petit", "court".into()))
        );
        let c_grand = cout!(
            store,
            store.persist_tombstone(&une_tombe("t-grand", "court".into()))
        );
        assert!(c_grand > c_petit, "MERGE : {c_grand} contre {c_petit}");
    }

    #[tokio::test]
    async fn update_sharing_policy_au_dela_du_seuil_et_son_remplacement_paient_le_flush_complet() {
        use project_orchestrator::episodes::distill_models::SharingPolicy;
        let (store, p) = store_avec_projet().await;
        let mut longue = SharingPolicy::default();
        longue.type_overrides.insert(
            "t".repeat(300),
            serde_json::from_value(serde_json::json!("never")).unwrap(),
        );
        let c_petit = cout!(
            store,
            store.update_sharing_policy(p.id, &SharingPolicy::default())
        );
        let c_grand = cout!(store, store.update_sharing_policy(p.id, &longue));
        assert!(c_grand > c_petit, "{c_grand} contre {c_petit}");
        let c_remplace = cout!(
            store,
            store.update_sharing_policy(p.id, &SharingPolicy::default())
        );
        assert!(
            c_remplace > c_petit,
            "remplacement : {c_remplace} contre {c_petit}"
        );
    }
}
