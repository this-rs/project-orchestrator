//! `SearchStore` on obrain (`src/obrain/search.rs`).
//!
//! Two kinds of test:
//! - the CONTRACT, run on the in-memory mock and on obrain with the same
//!   fixtures (single-word queries, where both engines must agree): filters,
//!   deletions, statistics, the default status of a note search, the status
//!   update. Meilisearch itself is not run here; the differences with it are
//!   listed in the module doc of `search.rs`.
//! - what only this backend does: the filter applied AFTER the text index (never a
//!   missing hit), freshness of the index after writes, a reopen, a torn write
//!   repaired at open, a batch cut before its commit, the named refusal of a query
//!   the index cannot answer, the score, and the refusal of `delete_project`.
#![cfg(feature = "obrain-backend")]

use std::path::Path;

use obrain_common::types::Value;
use project_orchestrator::meilisearch::indexes::{CodeDocument, DecisionDocument, NoteDocument};
use project_orchestrator::meilisearch::mock::MockSearchStore;
use project_orchestrator::meilisearch::{MeiliClient, SearchStore};
use project_orchestrator::obrain::{
    Creation, Interruption, NonPorte, ObrainGraphStore, MARQUEUR_PO, SCHEMA_PO,
};

async fn open(path: &Path) -> ObrainGraphStore {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ObrainGraphStore::open(path, Creation::Permise))
        .await
        .unwrap()
        .unwrap()
}

fn code(path: &str, slug: &str, lang: &str, symbols: &[&str]) -> CodeDocument {
    CodeDocument {
        id: MeiliClient::path_to_id(path),
        path: path.into(),
        language: lang.into(),
        symbols: symbols.iter().map(|s| s.to_string()).collect(),
        docstrings: String::new(),
        signatures: vec![],
        imports: vec![],
        project_id: format!("id-{slug}"),
        project_slug: slug.into(),
    }
}

fn decision(id: &str, text: &str, slug: Option<&str>, task: Option<&str>) -> DecisionDocument {
    DecisionDocument {
        id: id.into(),
        description: text.into(),
        rationale: String::new(),
        task_id: task.map(Into::into),
        agent: "a".into(),
        timestamp: "2026-11-20T10:00:00+00:00".into(),
        tags: vec![],
        project_id: slug.map(|s| format!("id-{s}")),
        project_slug: slug.map(Into::into),
    }
}

fn note(id: &str, content: &str, slug: &str, status: &str) -> NoteDocument {
    NoteDocument {
        id: id.into(),
        project_id: format!("id-{slug}"),
        project_slug: slug.into(),
        note_type: "gotcha".into(),
        status: status.into(),
        importance: "high".into(),
        scope_type: "project".into(),
        scope_path: String::new(),
        content: content.into(),
        tags: vec![],
        anchor_entities: vec![],
        created_at: 1_700_000_000,
        created_by: "t".into(),
        staleness_score: 0.0,
        author_kind: "agent".into(),
        human_confirmed: false,
        memory_horizon: "operational".into(),
    }
}

fn ids<T>(docs: &[T], id: impl Fn(&T) -> &str) -> Vec<String> {
    let mut v: Vec<String> = docs.iter().map(|d| id(d).to_string()).collect();
    v.sort();
    v
}

fn refusal<T: std::fmt::Debug>(r: anyhow::Result<T>) -> NonPorte {
    *r.expect_err("must refuse")
        .downcast_ref::<NonPorte>()
        .expect("a NonPorte, not another error")
}

// ===========================================================================
// The contract, on the mock and on obrain
// ===========================================================================

/// Code and decisions: run on the mock AND on obrain.
async fn contract_code_et_decisions(s: &dyn SearchStore) {
    assert!(s.health_check().await.unwrap());

    // ---- code
    s.index_code(&code("src/a.rs", "alpha", "rust", &["zork"]))
        .await
        .unwrap();
    s.index_code_batch(&[
        code("src/b.py", "alpha", "python", &["zork"]),
        code("lib/c.rs", "beta", "rust", &["zork"]),
        CodeDocument {
            project_id: String::new(),
            ..code("src/d.rs", "", "rust", &["zork"])
        },
    ])
    .await
    .unwrap();
    assert_eq!(s.get_code_stats().await.unwrap().total_documents, 4);
    let c = |v: Vec<CodeDocument>| ids(&v, |d| &d.path);
    assert_eq!(
        c(s.search_code("zork", 10, None).await.unwrap()),
        ["lib/c.rs", "src/a.rs", "src/b.py", "src/d.rs"]
    );
    assert_eq!(
        c(s.search_code("zork", 10, Some("rust")).await.unwrap()),
        ["lib/c.rs", "src/a.rs", "src/d.rs"]
    );
    assert_eq!(
        c(s.search_code_in_project("zork", 10, None, Some("alpha"))
            .await
            .unwrap()),
        ["src/a.rs", "src/b.py"]
    );
    assert_eq!(
        c(
            s.search_code_in_project("zork", 10, Some("rust"), Some("alpha"))
                .await
                .unwrap()
        ),
        ["src/a.rs"]
    );
    let hits = s
        .search_code_with_scores("zork", 10, None, Some("alpha"), Some("src/"))
        .await
        .unwrap();
    assert_eq!(
        c(hits.iter().map(|h| h.document.clone()).collect()),
        ["src/a.rs", "src/b.py"]
    );
    assert!(hits.iter().all(|h| h.score > 0.0 && h.score <= 1.0));
    assert!(s
        .search_code("zork", 10, Some("go"))
        .await
        .unwrap()
        .is_empty());
    assert_eq!(s.search_code("zork", 2, None).await.unwrap().len(), 2);
    // Indexing the same document again replaces it: still 4.
    s.index_code(&code("src/a.rs", "alpha", "rust", &["zork", "extra"]))
        .await
        .unwrap();
    assert_eq!(s.get_code_stats().await.unwrap().total_documents, 4);

    s.delete_code("src/b.py").await.unwrap();
    s.delete_code("never/indexed.rs").await.unwrap(); // the no-op of Meilisearch
    assert_eq!(s.get_code_stats().await.unwrap().total_documents, 3);
    s.delete_code_for_project("beta").await.unwrap();
    assert_eq!(s.get_code_stats().await.unwrap().total_documents, 2);
    s.delete_orphan_code_documents().await.unwrap();
    assert_eq!(
        c(s.search_code("zork", 10, None).await.unwrap()),
        ["src/a.rs"],
        "the document with an empty project slug/id is the orphan"
    );
    s.delete_all_code().await.unwrap();
    assert_eq!(s.get_code_stats().await.unwrap().total_documents, 0);

    // ---- decisions
    s.index_decision(&decision("d1", "quux alpha", Some("alpha"), Some("t1")))
        .await
        .unwrap();
    s.index_decision(&decision("d2", "quux beta", Some("beta"), Some("t2")))
        .await
        .unwrap();
    s.index_decision(&decision("d3", "quux free", None, None))
        .await
        .unwrap();
    let d = |v: Vec<DecisionDocument>| ids(&v, |d| &d.id);
    assert_eq!(
        d(s.search_decisions("quux", 10).await.unwrap()),
        ["d1", "d2", "d3"]
    );
    assert_eq!(
        d(s.search_decisions_in_project("quux", 10, Some("beta"))
            .await
            .unwrap()),
        ["d2"]
    );
    assert_eq!(
        d(s.search_decisions_in_project("quux", 10, None)
            .await
            .unwrap()),
        ["d1", "d2", "d3"]
    );
    assert_eq!(
        d(
            s.search_decisions_in_projects("quux", 10, &["alpha".into(), "beta".into()])
                .await
                .unwrap()
        ),
        ["d1", "d2"]
    );
    s.delete_decisions_for_task("t1").await.unwrap();
    assert_eq!(
        d(s.search_decisions("quux", 10).await.unwrap()),
        ["d2", "d3"]
    );
    s.delete_decisions_for_project("beta").await.unwrap();
    assert_eq!(d(s.search_decisions("quux", 10).await.unwrap()), ["d3"]);
    s.delete_decision("d3").await.unwrap();
    s.delete_decision("d3").await.unwrap();
    assert!(s.search_decisions("quux", 10).await.unwrap().is_empty());
}

/// Notes: run on obrain only. The mock does not apply the default status filter
/// (`active`, `needs_review`) that the Meilisearch adapter applies when no status
/// is given, so the mock answers differently on the same fixtures. The mock's
/// divergence is not fixed here; this suite pins the adapter's behaviour on obrain.
async fn contract_notes(s: &dyn SearchStore) {
    s.index_note(&note("n1", "frobnicate alpha", "alpha", "active"))
        .await
        .unwrap();
    s.index_notes_batch(&[
        note("n2", "frobnicate beta", "beta", "needs_review"),
        note("n3", "frobnicate stale", "alpha", "stale"),
        note("n4", "frobnicate archived", "alpha", "archived"),
    ])
    .await
    .unwrap();
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 4);
    let n = |v: Vec<NoteDocument>| ids(&v, |d| &d.id);
    // No status given: active and needs_review only.
    assert_eq!(
        n(s.search_notes("frobnicate", 10).await.unwrap()),
        ["n1", "n2"]
    );
    assert_eq!(
        n(
            s.search_notes_with_filters("frobnicate", 10, Some("alpha"), None, None, None)
                .await
                .unwrap()
        ),
        ["n1"]
    );
    assert_eq!(
        n(
            s.search_notes_with_filters("frobnicate", 10, None, None, Some("stale"), None)
                .await
                .unwrap()
        ),
        ["n3"]
    );
    assert_eq!(
        n(
            s.search_notes_with_filters("frobnicate", 10, None, Some("tip"), None, None)
                .await
                .unwrap()
        ),
        Vec::<String>::new()
    );
    assert_eq!(
        n(
            s.search_notes_with_filters("frobnicate", 10, None, None, None, Some("high"))
                .await
                .unwrap()
        ),
        ["n1", "n2"]
    );
    let hits = s
        .search_notes_with_scores("frobnicate", 10, None, Some("gotcha"), None, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    s.update_note_status("n1", "archived").await.unwrap();
    s.update_note_status("absent", "archived").await.unwrap();
    assert_eq!(n(s.search_notes("frobnicate", 10).await.unwrap()), ["n2"]);
    s.delete_notes_for_project("beta").await.unwrap();
    assert!(s.search_notes("frobnicate", 10).await.unwrap().is_empty());
    s.delete_note("n3").await.unwrap();
    // n1 (archived by status update, still indexed) and n4 remain.
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 2);
    assert!(!s.get_notes_stats().await.unwrap().is_indexing);
}

#[tokio::test]
async fn code_and_decisions_hold_on_the_mock() {
    contract_code_et_decisions(&MockSearchStore::new()).await;
}

#[tokio::test]
async fn code_and_decisions_hold_on_obrain() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    contract_code_et_decisions(&g.recherche()).await;
}

#[tokio::test]
async fn notes_hold_on_obrain() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    contract_notes(&g.recherche()).await;
}

// ===========================================================================
// What only this backend does
// ===========================================================================

/// The filter is applied after `text_search`, which takes none: a project whose
/// documents all rank BELOW the first 32 BM25 hits must still be found. REGRESSION:
/// without the widening of `k`, this returns nothing for `beta`.
#[tokio::test]
async fn a_selective_filter_finds_documents_ranked_below_the_first_hits() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    let mut docs = Vec::new();
    // 80 documents of `alpha` that repeat the term and are short: they outrank
    // the 3 of `beta`, which say it once among other words.
    for i in 0..80 {
        docs.push(code(
            &format!("a/{i}.rs"),
            "alpha",
            "rust",
            &["zork", "zork", "zork"],
        ));
    }
    for i in 0..3 {
        docs.push(CodeDocument {
            docstrings: "una frase larga con muchas otras palabras que diluyen el termino".into(),
            ..code(&format!("b/{i}.rs"), "beta", "rust", &["zork"])
        });
    }
    s.index_code_batch(&docs).await.unwrap();
    let hits = s
        .search_code_in_project("zork", 3, None, Some("beta"))
        .await
        .unwrap();
    assert_eq!(
        ids(&hits, |d| &d.path),
        ["b/0.rs", "b/1.rs", "b/2.rs"],
        "all three, though none is in the first 32 hits of the ranking"
    );
    // And a filter that matches nothing terminates with nothing, not with alpha.
    assert!(s
        .search_code_in_project("zork", 3, None, Some("gamma"))
        .await
        .unwrap()
        .is_empty());
    // An unfiltered search still returns the best, `alpha`.
    let top = s.search_code("zork", 3, None).await.unwrap();
    assert!(top.iter().all(|d| d.project_slug == "alpha"));
}

/// The text index follows writes made after it was built: a new document is found,
/// a replaced one is found by its NEW text only, a deleted one is gone.
#[tokio::test]
async fn the_text_index_follows_writes_after_its_first_build() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    s.index_note(&note("n1", "wombat", "p", "active"))
        .await
        .unwrap();
    assert_eq!(s.search_notes("wombat", 5).await.unwrap().len(), 1); // builds the index
    s.index_note(&note("n2", "wombat quokka", "p", "active"))
        .await
        .unwrap();
    assert_eq!(s.search_notes("quokka", 5).await.unwrap().len(), 1);
    s.index_note(&note("n2", "numbat", "p", "active"))
        .await
        .unwrap();
    assert!(s.search_notes("quokka", 5).await.unwrap().is_empty());
    assert_eq!(s.search_notes("numbat", 5).await.unwrap().len(), 1);
    // The RAW index, not only the served answer: the replaced text is gone from it.
    assert_eq!(s.bm25_brut("RechercheNote", "quokka", 10).unwrap(), 0);
    s.delete_note("n1").await.unwrap();
    assert!(s.search_notes("wombat", 5).await.unwrap().is_empty());
    // A deleted document leaves the engine AND its text index: a search that only
    // filtered it out would pass the assertion above and fail this one.
    assert_eq!(s.bm25_brut("RechercheNote", "wombat", 10).unwrap(), 0);
    assert_eq!(s.noeuds_du_label("RechercheNote"), 1);
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 1);
}

/// A reopen: documents, and the ability to search them, survive.
#[tokio::test]
async fn documents_survive_a_close_and_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("b");
    let g = open(&base).await;
    let s = g.recherche();
    s.index_note(&note("n1", &"wombat ".repeat(100), "p", "active"))
        .await
        .unwrap();
    s.index_code(&code("src/x.rs", "p", "rust", &["marsupial"]))
        .await
        .unwrap();
    s.index_decision(&decision("d1", "cassowary", Some("p"), None))
        .await
        .unwrap();
    assert_eq!(s.search_notes("wombat", 5).await.unwrap().len(), 1);
    g.close().await.unwrap();
    drop((s, g));

    let g = open(&base).await;
    assert_eq!(*g.rapport_d_ouverture(), Default::default());
    let s = g.recherche();
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 1);
    assert_eq!(s.search_notes("wombat", 5).await.unwrap()[0].id, "n1");
    assert_eq!(s.search_code("marsupial", 5, None).await.unwrap().len(), 1);
    assert_eq!(s.search_decisions("cassowary", 5).await.unwrap().len(), 1);
    // The reopened base keeps taking writes.
    s.index_note(&note("n2", "wombat", "p", "active"))
        .await
        .unwrap();
    assert_eq!(s.search_notes("wombat", 5).await.unwrap().len(), 2);
}

/// Write straight into a closed base, as an interrupted write would have left it.
async fn write_raw(base: &Path, f: impl FnOnce(&obrain_engine::ObrainDB) + Send + 'static) {
    let base = base.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let db = obrain_engine::ObrainDB::open(&base).unwrap();
        f(&db);
        db.commit_wal().unwrap();
        db.close().unwrap();
        std::fs::write(base.join(MARQUEUR_PO), SCHEMA_PO).unwrap();
    })
    .await
    .unwrap();
}

/// An update cut between the write of `doc` and the write of `texte`: the document
/// says one thing, the text index would be built from another. Open recomputes
/// `texte` from `doc`. REGRESSION: without the repair, searching for the NEW word
/// finds nothing and the OLD word finds a document that no longer says it.
#[tokio::test]
async fn a_torn_update_is_repaired_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("b");
    let g = open(&base).await;
    g.recherche()
        .index_note(&note("n1", "oldword", "p", "active"))
        .await
        .unwrap();
    g.close().await.unwrap();
    drop(g);

    // `doc` rewritten, `texte` left behind.
    write_raw(&base, |db| {
        let noeud = db.store().nodes_by_label("RechercheNote")[0];
        let json = serde_json::to_string(&note("n1", "newword", "p", "active")).unwrap();
        db.set_node_property(noeud, "doc", Value::from(json));
    })
    .await;

    let g = open(&base).await;
    assert_eq!(g.rapport_d_ouverture().recherches_reparees, 1);
    let s = g.recherche();
    assert_eq!(s.search_notes("newword", 5).await.unwrap().len(), 1);
    assert!(s.search_notes("oldword", 5).await.unwrap().is_empty());
    g.close().await.unwrap();
    drop((s, g));
    // Repaired once, for good.
    let g = open(&base).await;
    assert_eq!(g.rapport_d_ouverture().recherches_reparees, 0);
}

/// A batch cut after its nodes are written and before their markers are removed
/// leaves NONE of its new documents, and open counts the drafts it deleted.
#[tokio::test]
async fn a_batch_cut_before_its_commit_leaves_none_of_its_new_documents() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("b");
    let g = open(&base).await;
    let s = g.recherche();
    s.index_note(&note("witness", "wombat", "p", "active"))
        .await
        .unwrap();
    g.interrompre_au_prochain(Interruption::AvantLeRetraitDuMarqueur);
    let err = s
        .index_notes_batch(&[
            note("b1", "wombat", "p", "active"),
            note("b2", "wombat", "p", "active"),
            note("b3", "wombat", "p", "active"),
        ])
        .await
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("interruption injectée"),
        "{err:#}"
    );
    // Nothing published.
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 1);
    g.close().await.unwrap();
    drop((s, g));

    let g = open(&base).await;
    assert_eq!(g.rapport_d_ouverture().brouillons_supprimes, 3);
    let s = g.recherche();
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 1);
    assert_eq!(
        ids(&s.search_notes("wombat", 10).await.unwrap(), |d| &d.id),
        ["witness"]
    );
    // The retry succeeds.
    s.index_notes_batch(&[note("b1", "wombat", "p", "active")])
        .await
        .unwrap();
    assert_eq!(s.search_notes("wombat", 10).await.unwrap().len(), 2);
}

/// A query the index cannot answer is refused BY NAME, never answered empty; the
/// placeholder search (no alphanumeric) is answered with the documents.
#[tokio::test]
async fn a_query_the_index_cannot_answer_is_refused_and_the_placeholder_is_served() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    s.index_note(&note("n2", "the one", "p", "active"))
        .await
        .unwrap();
    s.index_note(&note("n1", "the other", "p", "active"))
        .await
        .unwrap();
    for q in ["the", "how are you", "a b"] {
        assert_eq!(
            refusal(
                s.search_notes_with_scores(q, 5, None, None, None, None)
                    .await
            ),
            NonPorte("SearchStore::search_notes_with_scores"),
            "{q}"
        );
    }
    // The refusal does not depend on the corpus: an empty kind refuses too.
    assert_eq!(
        refusal(s.search_code_with_scores("the", 5, None, None, None).await),
        NonPorte("SearchStore::search_code_with_scores")
    );
    // Placeholder: all documents that pass the filters, by increasing id.
    for q in ["", "*", "  ", "?!"] {
        let hits = s
            .search_notes_with_scores(q, 5, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|h| h.document.id.as_str())
                .collect::<Vec<_>>(),
            ["n1", "n2"],
            "{q:?}"
        );
        assert!(hits.iter().all(|h| h.score == 1.0));
    }
    assert!(s.search_notes("", 0).await.unwrap().is_empty());
}

/// `score` is driven by the fraction of the query's terms the document carries:
/// 1.0 for a duplicate (what `chat/feedback.rs` thresholds at 0.85), less for a
/// partial match.
#[tokio::test]
async fn the_score_is_the_fraction_of_query_terms_found() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    s.index_note(&note(
        "dup",
        "migration obrain durable fsync journal",
        "p",
        "active",
    ))
    .await
    .unwrap();
    s.index_note(&note("part", "migration neo4j", "p", "active"))
        .await
        .unwrap();
    let hits = s
        .search_notes_with_scores(
            "migration obrain durable fsync journal",
            5,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(hits[0].document.id, "dup");
    // Full coverage: (1 − δ) + δ·b/(1+b), so in [0.999, 1).
    assert!(
        hits[0].score >= 0.999 && hits[0].score < 1.0,
        "{}",
        hits[0].score
    );
    let part = hits.iter().find(|h| h.document.id == "part").unwrap();
    // One term of five: within δ = 10⁻³ of 0.2.
    assert!((part.score - 0.2).abs() <= 1e-3, "{}", part.score);
}

/// A document without an id is refused with nothing written; a re-indexed id is
/// replaced, not duplicated.
#[tokio::test]
async fn an_id_is_required_and_replaces() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    let err = s.index_note(&note("", "wombat", "p", "active")).await;
    assert!(err.is_err());
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 0);
    s.index_note(&note("n1", "wombat", "p", "active"))
        .await
        .unwrap();
    s.index_note(&note("n1", "wombat again", "p", "active"))
        .await
        .unwrap();
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 1);
    // A batch with the same id twice: the last wins.
    s.index_notes_batch(&[
        note("n2", "first", "p", "active"),
        note("n2", "second", "p", "active"),
    ])
    .await
    .unwrap();
    assert_eq!(s.get_notes_stats().await.unwrap().total_documents, 2);
    assert!(s.search_notes("first", 5).await.unwrap().is_empty());
    assert_eq!(s.search_notes("second", 5).await.unwrap().len(), 1);
}

/// The search documents are nodes of the base: `delete_project` refuses while any
/// exists, and its refusal NAMES them under « recherche ».
#[tokio::test]
async fn delete_project_refusal_names_the_search_documents() {
    use project_orchestrator::neo4j::models::{ProjectNode, ProjectProfile};
    use project_orchestrator::neo4j::traits::ProjectStore;
    use uuid::Uuid;
    let g = ObrainGraphStore::open_temporary().unwrap();
    let id = Uuid::new_v4();
    let p = ProjectNode {
        id,
        name: "p".into(),
        slug: format!("p-{id}"),
        root_path: String::new(),
        description: Some("d".into()),
        created_at: chrono::Utc::now(),
        last_synced: None,
        analytics_computed_at: None,
        last_co_change_computed_at: None,
        default_note_energy: None,
        scaffolding_override: None,
        sharing_policy: None,
        watch_enabled: false,
        profile: ProjectProfile::Software,
    };
    g.create_project(&p).await.unwrap();
    g.recherche()
        .index_note(&note("n1", "wombat", "p", "active"))
        .await
        .unwrap();
    let err = format!("{:#}", g.delete_project(id, "p").await.unwrap_err());
    assert!(
        err.contains("recherche (RechercheNote ×1)"),
        "the refusal names the search documents: {err}"
    );
}

// ===========================================================================
// Review of #1251
// ===========================================================================

/// (1) Writes that run WHILE the text index is first built are all in it. The index
/// is built at open, before any writer; a build at search time takes the writer
/// lock. REGRESSION: a lazy build under the read lock missed the writes that landed
/// between its scan and its registration, for good.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_concurrent_with_the_first_search_are_all_indexed() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = std::sync::Arc::new(g.recherche());
    let lot: Vec<NoteDocument> = (0..20_000)
        .map(|i| note(&format!("a{i:05}"), "wombat", "p", "active"))
        .collect();
    s.index_notes_batch(&lot).await.unwrap();
    let ecrivain = {
        let s = s.clone();
        tokio::spawn(async move {
            for i in 0..200 {
                s.index_note(&note(&format!("b{i:03}"), "wombat", "p", "active"))
                    .await
                    .unwrap();
            }
        })
    };
    // The first search, racing the writer.
    let _ = s.search_notes("wombat", 1).await.unwrap();
    ecrivain.await.unwrap();
    assert_eq!(
        s.bm25_brut("RechercheNote", "wombat", 100_000).unwrap(),
        20_200,
        "every document written, before or during the first search, is in the index"
    );
}

/// (2) Nodes the index holds but that are not served (a batch cut before its
/// commit, still in the engine until the next open) and that outrank every served
/// document must not exhaust the search. REGRESSION: the budget was capped at the
/// number of SERVED documents, so the answer came back empty.
#[tokio::test]
async fn unserved_nodes_ranked_first_do_not_empty_the_answer() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    for i in 0..3 {
        s.index_note(&note(
            &format!("ok{i}"),
            "zork among a long sentence of many other unrelated filler words here",
            "p",
            "active",
        ))
        .await
        .unwrap();
    }
    g.interrompre_au_prochain(Interruption::AvantLeRetraitDuMarqueur);
    let brouillons: Vec<NoteDocument> = (0..40)
        .map(|i| note(&format!("draft{i}"), "zork zork zork", "p", "active"))
        .collect();
    assert!(s.index_notes_batch(&brouillons).await.is_err());
    assert_eq!(s.bm25_brut("RechercheNote", "zork", 100).unwrap(), 43);
    let hits = s.search_notes("zork", 3).await.unwrap();
    assert_eq!(ids(&hits, |d| &d.id), ["ok0", "ok1", "ok2"]);
}

/// (4) A batch cut at ANY step leaves all of its new documents or none, never a
/// prefix. REGRESSION: each document was committed by its own marker removal, so
/// a cut between two of them left the first ones.
#[tokio::test]
async fn a_batch_cut_at_any_step_leaves_all_or_none_of_its_documents() {
    let dir = tempfile::tempdir().unwrap();
    for n in 1..=6 {
        let base = dir.path().join(format!("b{n}"));
        let g = open(&base).await;
        let s = g.recherche();
        s.index_note(&note("witness", "wombat", "p", "active"))
            .await
            .unwrap();
        g.interrompre_au_prochain(Interruption::AvantEtape(n));
        let _ = s
            .index_notes_batch(&[
                note("b1", "wombat", "p", "active"),
                note("b2", "wombat", "p", "active"),
                note("b3", "wombat", "p", "active"),
            ])
            .await;
        g.close().await.unwrap();
        drop((s, g));
        let g = open(&base).await;
        let s = g.recherche();
        let total = s.get_notes_stats().await.unwrap().total_documents;
        assert!(
            total == 1 || total == 4,
            "cut before step {n}: {total} documents, a prefix of the batch survived"
        );
        assert_eq!(s.noeuds_du_label("RechercheLot"), 0, "step {n}");
        g.close().await.unwrap();
    }
}

/// (6) A document being rewritten (its `doc` written, the rest not yet) is never
/// returned with fields that contradict the filter. REGRESSION: the filter read the
/// published copy (project `alpha`) and returned the document read (project `beta`).
#[tokio::test]
async fn a_document_mid_rewrite_never_contradicts_the_filter() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    s.index_note(&note("n1", "wombat", "alpha", "active"))
        .await
        .unwrap();
    // Steps of a replacement: 1 before the document, 2 between `doc` and `texte`.
    g.interrompre_au_prochain(Interruption::AvantEtape(2));
    assert!(s
        .index_note(&note("n1", "wombat", "beta", "active"))
        .await
        .is_err());
    for slug in ["alpha", "beta"] {
        let hits = s
            .search_notes_with_filters("wombat", 5, Some(slug), None, None, None)
            .await
            .unwrap();
        assert!(
            hits.iter().all(|d| d.project_slug == slug),
            "filter {slug}, returned {:?}",
            hits.iter().map(|d| &d.project_slug).collect::<Vec<_>>()
        );
    }
}

/// (7) Hits come in decreasing score; a duplicate of the query comes first even
/// when another document outranks it in BM25 (by repeating one rare term), with a
/// score above the 0.85 of `chat/feedback.rs`. REGRESSION: hits were in BM25 order
/// with a score that was not monotone in it, so the first hit was not the duplicate.
#[tokio::test]
async fn the_score_orders_the_hits_and_puts_the_duplicate_first() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    let mut docs = vec![
        note("spam", "alpha alpha alpha alpha alpha alpha", "p", "active"),
        note(
            "dup",
            &format!(
                "alpha beta gamma delta epsilon {}",
                "lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(6)
            ),
            "p",
            "active",
        ),
    ];
    for i in 0..30 {
        docs.push(note(
            &format!("f{i:02}"),
            "beta gamma delta epsilon",
            "p",
            "active",
        ));
    }
    s.index_notes_batch(&docs).await.unwrap();
    let hits = s
        .search_notes_with_scores("alpha beta gamma delta epsilon", 10, None, None, None, None)
        .await
        .unwrap();
    assert_eq!(
        hits[0].document.id,
        "dup",
        "{:?}",
        hits.iter()
            .map(|h| (&h.document.id, h.score))
            .collect::<Vec<_>>()
    );
    assert!(hits[0].score > 0.85, "{}", hits[0].score);
    assert!(
        hits.windows(2).all(|w| w[0].score >= w[1].score),
        "decreasing score: {:?}",
        hits.iter().map(|h| h.score).collect::<Vec<_>>()
    );
}

/// (7) Lists of several projects merged by score (`api/code_handlers.rs`) give the
/// order of the single search over all projects.
#[tokio::test]
async fn per_project_lists_merged_by_score_give_the_global_order() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    let mut docs = Vec::new();
    for (i, texte) in [
        "zork",
        "zork zork quux",
        "zork plugh",
        "zork zork zork",
        "quux zork xyzzy plugh",
        "zork xyzzy",
    ]
    .iter()
    .enumerate()
    {
        let slug = if i % 2 == 0 { "alpha" } else { "beta" };
        docs.push(CodeDocument {
            docstrings: (*texte).into(),
            ..code(&format!("{slug}/{i}.rs"), slug, "rust", &[])
        });
    }
    s.index_code_batch(&docs).await.unwrap();
    let q = "zork quux plugh";
    let global: Vec<String> = s
        .search_code_with_scores(q, 10, None, None, None)
        .await
        .unwrap()
        .into_iter()
        .map(|h| h.document.path)
        .collect();
    let mut fusion = Vec::new();
    for slug in ["alpha", "beta"] {
        fusion.extend(
            s.search_code_with_scores(q, 10, None, Some(slug), None)
                .await
                .unwrap(),
        );
    }
    fusion.sort_by(|a, b| b.score.total_cmp(&a.score));
    let fusion: Vec<String> = fusion.into_iter().map(|h| h.document.path).collect();
    assert_eq!(fusion, global);
}

/// (8) Quarantined nodes go with the deletions that cover what can be read of them,
/// and indexing their id again replaces them. REGRESSION: they were never deleted
/// and refused the re-indexing of their id.
#[tokio::test]
async fn quarantined_nodes_are_deleted_and_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("b");
    let g = open(&base).await;
    g.close().await.unwrap();
    drop(g);
    write_raw(&base, |db| {
        let poser = |label: &str, id: &str, doc: &str| {
            let n = db.create_node(&[label]);
            db.set_node_property(n, "id", Value::from(id.to_string()));
            db.set_node_property(n, "doc", Value::from(doc.to_string()));
        };
        // Readable JSON, not a document: its project can be read.
        poser(
            "RechercheCode",
            "q1",
            r#"{"project_slug":"alpha","project_id":"x"}"#,
        );
        // Not JSON at all: no readable project, an orphan.
        poser("RechercheCode", "q2", "not json");
        // A note whose id will be indexed again.
        poser("RechercheNote", "qn", "not json either");
    })
    .await;
    let g = open(&base).await;
    assert_eq!(g.rapport_d_ouverture().en_quarantaine.len(), 3);
    let s = g.recherche();
    assert_eq!(s.noeuds_du_label("RechercheCode"), 2);
    s.delete_code_for_project("alpha").await.unwrap();
    assert_eq!(
        s.noeuds_du_label("RechercheCode"),
        1,
        "q1 went with its project"
    );
    s.delete_orphan_code_documents().await.unwrap();
    assert_eq!(s.noeuds_du_label("RechercheCode"), 0, "q2 is an orphan");
    s.index_note(&note("qn", "wombat", "p", "active"))
        .await
        .unwrap();
    assert_eq!(
        s.noeuds_du_label("RechercheNote"),
        1,
        "replaced, not doubled"
    );
    assert_eq!(s.search_notes("wombat", 5).await.unwrap()[0].id, "qn");
    g.close().await.unwrap();
    drop((s, g));
    let g = open(&base).await;
    assert!(g.rapport_d_ouverture().en_quarantaine.is_empty());
}

/// (5) `search_decisions_in_projects(&[])` matches nothing, without a query check.
#[tokio::test]
async fn no_project_means_no_decision() {
    let g = ObrainGraphStore::open_temporary().unwrap();
    let s = g.recherche();
    s.index_decision(&decision("d1", "quux", Some("p"), None))
        .await
        .unwrap();
    assert!(s
        .search_decisions_in_projects("quux", 10, &[])
        .await
        .unwrap()
        .is_empty());
}
