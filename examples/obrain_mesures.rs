//! Cost of the obrain backend on a base of N nodes — the numbers behind the
//! claims of `src/obrain/`.
//!
//! ```text
//! cargo run --release --features obrain-backend,mock-store --example obrain_mesures -- <nodes> <edges> [--herite]
//! ```
//!
//! `--herite` sets `OBRAIN_PROPS_V2=0` before the base is created: the engine
//! then keeps node properties in its inherited format (a `DashMap` persisted by
//! a full rewrite of the sidecar at every flush) instead of the property zone.
//!
//! The base lives in a temporary directory this program creates and removes.
//! It reads no configuration and opens nothing else.
//!
//! Output: one `mesure<TAB>valeur<TAB>unité` line per measure.

use project_orchestrator::neo4j::models::{ProjectNode, ProjectProfile};
use project_orchestrator::neo4j::traits::{NoteStore, ProjectStore};
use project_orchestrator::notes::{EntityType, Note, NoteFilters, NoteType};
use project_orchestrator::obrain::{Creation, Durabilite, ObrainGraphStore};
use std::time::{Duration, Instant};
use uuid::Uuid;

fn projet(i: usize) -> ProjectNode {
    let id = Uuid::new_v4();
    ProjectNode {
        id,
        name: format!("mesure {i:05}"),
        slug: format!("mesure-{id}"),
        root_path: String::new(),
        description: Some(format!("projet de mesure {i}")),
        created_at: chrono::Utc::now(),
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

fn pct(mut v: Vec<Duration>, p: f64) -> Duration {
    v.sort();
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn dire(nom: &str, valeur: f64, unite: &str) {
    println!("{nom}\t{valeur:.1}\t{unite}");
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let noeuds: u64 = args.first().and_then(|a| a.parse().ok()).expect("<nodes>");
    let aretes: u64 = args.get(1).and_then(|a| a.parse().ok()).expect("<edges>");
    if args.iter().any(|a| a == "--herite") {
        std::env::set_var("OBRAIN_PROPS_V2", "0");
    }
    println!(
        "base\t{noeuds} nœuds, {aretes} arêtes, format {}",
        if args.iter().any(|a| a == "--herite") {
            "hérité (OBRAIN_PROPS_V2=0)"
        } else {
            "neuf"
        }
    );

    let dir = tempfile::Builder::new()
        .prefix("po-obrain-mesures-")
        .tempdir()
        .unwrap();
    let base = dir.path().join("base");

    // 1. Build the base.
    let store = ObrainGraphStore::open(&base, Creation::Permise).unwrap();
    let rempli = store.remplir_pour_mesure(noeuds, aretes).unwrap();
    dire("remplissage", rempli.as_secs_f64(), "s");
    store.close().await.unwrap();
    drop(store);

    // 2. Open it: the scan that rebuilds the indexes.
    let t = Instant::now();
    let store = ObrainGraphStore::open_avec(&base, Creation::Interdite, Durabilite::Wal).unwrap();
    dire("ouverture", t.elapsed().as_secs_f64() * 1e3, "ms");
    dire("noeuds_vivants", store.noeuds_pour_mesure() as f64, "nœuds");

    // The fill left a great deal of dirty data for the system to write back; an fsync
    // measured behind it measures the write-back, not the write. Let it settle first.
    let _ = std::process::Command::new("sync").status();
    tokio::time::sleep(Duration::from_secs(3)).await;

    // 3. Writes, WAL fsync per write (the production step).
    let mut mesures = Vec::new();
    let mut ids = Vec::new();
    for i in 0..200 {
        let p = projet(i);
        ids.push((p.id, p.slug.clone()));
        let t = Instant::now();
        store.create_project(&p).await.unwrap();
        mesures.push(t.elapsed());
    }
    dire("create_wal_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("create_wal_p99", us(pct(mesures.clone(), 0.99)), "µs");
    dire(
        "create_wal_moy",
        us(mesures.iter().sum::<Duration>() / mesures.len() as u32),
        "µs",
    );
    let mut mesures = Vec::new();
    for (id, _) in ids.iter().take(100) {
        let t = Instant::now();
        store
            .update_project(*id, Some(format!("renomme {id}")), None, None, None)
            .await
            .unwrap();
        mesures.push(t.elapsed());
    }
    dire("update_wal_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("update_wal_p99", us(pct(mesures, 0.99)), "µs");

    // A write that carries a value past the blob threshold pays the full flush too.
    let mut mesures = Vec::new();
    for i in 0..30 {
        let mut p = projet(500 + i);
        p.description = Some("é-long ".repeat(60));
        let t = Instant::now();
        store.create_project(&p).await.unwrap();
        mesures.push(t.elapsed());
    }
    dire("create_blob_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("create_blob_p99", us(pct(mesures, 0.99)), "µs");

    // 4. Reads.
    let rondes = 200_000u32;
    let t = Instant::now();
    for k in 0..rondes {
        let (id, _) = &ids[(k as usize) % ids.len()];
        std::hint::black_box(store.get_project(*id).await.unwrap());
    }
    dire(
        "get_project",
        t.elapsed().as_secs_f64() * 1e9 / rondes as f64,
        "ns",
    );
    let t = Instant::now();
    for k in 0..rondes {
        let (_, slug) = &ids[(k as usize) % ids.len()];
        std::hint::black_box(store.get_project_by_slug(slug).await.unwrap());
    }
    dire(
        "get_project_by_slug",
        t.elapsed().as_secs_f64() * 1e9 / rondes as f64,
        "ns",
    );
    let t = Instant::now();
    for k in 0..rondes {
        let (id, _) = &ids[(k as usize) % ids.len()];
        std::hint::black_box(store.path_policy(*id).await.unwrap());
    }
    dire(
        "path_policy",
        t.elapsed().as_secs_f64() * 1e9 / rondes as f64,
        "ns",
    );
    let t = Instant::now();
    let liste = store.list_projects().await.unwrap();
    dire("list_projects_230", us(t.elapsed()), "µs");
    assert_eq!(liste.len(), 230);

    // 5. Delete: refused on a base with other data (one counter), and the cost of
    //    the refusal.
    let t = Instant::now();
    let refus = store.delete_project(ids[0].0, "x").await;
    dire("delete_refuse", us(t.elapsed()), "µs");
    assert_eq!(refus.is_err(), noeuds > 0);

    // 5b. The note domain (NoteStore) on the same base: a project of notes, short and
    //     long contents (a value past the blob threshold pays the full flush), reads by
    //     id, by project, over the whole base, and by linked entity.
    let proj = ids[0].0;
    let mut mesures = Vec::new();
    let mut notes_ids = Vec::new();
    for i in 0..200 {
        let n = Note::new(
            Some(proj),
            NoteType::Observation,
            format!("note de mesure {i}, courte"),
            "mesure".to_string(),
        );
        let t = Instant::now();
        store.create_note(&n).await.unwrap();
        mesures.push(t.elapsed());
        notes_ids.push(n.id);
    }
    dire("create_note_court_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("create_note_court_p99", us(pct(mesures, 0.99)), "µs");
    let mut mesures = Vec::new();
    for i in 0..30 {
        let n = Note::new(
            Some(proj),
            NoteType::Observation,
            format!("note longue {i} {}", "é-long ".repeat(60)),
            "mesure".to_string(),
        );
        let t = Instant::now();
        store.create_note(&n).await.unwrap();
        mesures.push(t.elapsed());
        notes_ids.push(n.id);
    }
    dire("create_note_long_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("create_note_long_p99", us(pct(mesures, 0.99)), "µs");
    let mut mesures = Vec::new();
    for id in notes_ids.iter().take(100) {
        let t = Instant::now();
        store
            .update_note(
                *id,
                Some("contenu mis à jour".into()),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        mesures.push(t.elapsed());
    }
    dire("update_note_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("update_note_p99", us(pct(mesures, 0.99)), "µs");
    let rondes = 200_000u32;
    let t = Instant::now();
    for k in 0..rondes {
        let id = notes_ids[(k as usize) % notes_ids.len()];
        std::hint::black_box(store.get_note(id).await.unwrap());
    }
    dire(
        "get_note",
        t.elapsed().as_secs_f64() * 1e9 / rondes as f64,
        "ns",
    );
    let filtre = NoteFilters {
        limit: Some(20),
        ..Default::default()
    };
    let t = Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(store.list_notes(Some(proj), None, &filtre).await.unwrap());
    }
    dire("list_notes_projet", us(t.elapsed()) / 1000.0, "µs");
    let t = Instant::now();
    for _ in 0..20 {
        std::hint::black_box(store.list_notes(None, None, &filtre).await.unwrap());
    }
    dire("list_notes_toutes", us(t.elapsed()) / 20.0, "µs");
    let mut mesures = Vec::new();
    for id in notes_ids.iter().take(100) {
        let t = Instant::now();
        store
            .link_note_to_entity(
                *id,
                &EntityType::Project,
                &proj.to_string(),
                Some("s"),
                Some("b"),
            )
            .await
            .unwrap();
        mesures.push(t.elapsed());
    }
    dire("link_note_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("link_note_p99", us(pct(mesures, 0.99)), "µs");
    let t = Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(
            store
                .get_notes_for_entity(&EntityType::Project, &proj.to_string())
                .await
                .unwrap(),
        );
    }
    dire("get_notes_for_entity", us(t.elapsed()) / 1000.0, "µs");

    // 6. The first version's step: a full flush after EVERY write.
    store.close().await.unwrap();
    drop(store);
    let store =
        ObrainGraphStore::open_avec(&base, Creation::Interdite, Durabilite::Complet).unwrap();
    let mut mesures = Vec::new();
    let essais = if noeuds >= 1_000_000 { 10 } else { 50 };
    for i in 0..essais {
        let p = projet(1000 + i);
        let t = Instant::now();
        store.create_project(&p).await.unwrap();
        mesures.push(t.elapsed());
    }
    dire(
        "create_flush_complet_p50",
        us(pct(mesures.clone(), 0.5)),
        "µs",
    );
    dire("create_flush_complet_max", us(pct(mesures, 1.0)), "µs");
    store.close().await.unwrap();
}
