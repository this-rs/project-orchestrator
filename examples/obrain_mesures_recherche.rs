//! Cost of `SearchStore` on the obrain backend at N documents — the numbers behind
//! the "Cost" section of `src/obrain/search.rs`.
//!
//! ```text
//! cargo run --release --features obrain-backend,mock-store --example obrain_mesures_recherche -- <documents>
//! ```
//!
//! `<documents>` code documents are indexed (batches of 10 000), spread over 100
//! projects. Then: the close and the reopen (which rebuilds the in-memory map and
//! the text index from the documents), the first search after the reopen, a
//! filtered search, the placeholder search without and with a selective filter,
//! and `delete_code` by id. Median and p99 over the runs, in microseconds.
//!
//! The base lives in a temporary directory this program creates and removes.
//! Output: one `mesure<TAB>valeur<TAB>unité` line per measure.

use project_orchestrator::meilisearch::indexes::CodeDocument;
use project_orchestrator::meilisearch::{MeiliClient, SearchStore};
use project_orchestrator::obrain::{Creation, ObrainGraphStore};
use std::time::{Duration, Instant};

const PROJETS: usize = 100;
const LOT: usize = 10_000;

fn document(i: usize) -> CodeDocument {
    let path = format!("src/module_{i:07}.rs");
    let p = i % PROJETS;
    CodeDocument {
        id: MeiliClient::path_to_id(&path),
        path,
        language: "rust".into(),
        // A word every document carries, one shared by 1 %, one unique.
        symbols: vec![
            "commun".into(),
            format!("groupe{}", i % 100),
            format!("unique{i}"),
        ],
        docstrings: "un texte de documentation ordinaire pour la mesure".into(),
        signatures: vec![],
        imports: vec![],
        project_id: format!("id-{p}"),
        project_slug: format!("projet-{p}"),
    }
}

fn sortie(mesure: &str, valeur: f64, unite: &str) {
    println!("{mesure}\t{valeur:.1}\t{unite}");
}

fn quantiles(mesure: &str, mut d: Vec<Duration>) {
    d.sort();
    let us = |x: Duration| x.as_secs_f64() * 1e6;
    sortie(&format!("{mesure}_p50"), us(d[d.len() / 2]), "us");
    sortie(
        &format!("{mesure}_p99"),
        us(d[(d.len() * 99 / 100).min(d.len() - 1)]),
        "us",
    );
}

fn charge() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "inconnue".into())
}

#[tokio::main]
async fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .expect("usage: obrain_mesures_recherche <documents>");
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base");
    println!("documents\t{n}\t-");
    println!("charge_debut\t{}\t-", charge());

    let g = ObrainGraphStore::open(&base, Creation::Permise).unwrap();
    let s = g.recherche();
    let t0 = Instant::now();
    let mut i = 0;
    while i < n {
        let fin = (i + LOT).min(n);
        let lot: Vec<CodeDocument> = (i..fin).map(document).collect();
        s.index_code_batch(&lot).await.unwrap();
        i = fin;
    }
    sortie("indexation_totale", t0.elapsed().as_secs_f64() * 1e3, "ms");
    g.close().await.unwrap();
    drop((s, g));

    let t0 = Instant::now();
    let g = ObrainGraphStore::open(&base, Creation::Interdite).unwrap();
    sortie("ouverture", t0.elapsed().as_secs_f64() * 1e3, "ms");
    let s = g.recherche();
    let t0 = Instant::now();
    let premiere = s.search_code("commun", 10, None).await.unwrap();
    sortie(
        "premiere_recherche_apres_ouverture",
        t0.elapsed().as_secs_f64() * 1e6,
        "us",
    );
    assert_eq!(premiere.len(), 10);

    let runs = 200;
    let mut d = Vec::with_capacity(runs);
    for k in 0..runs {
        let t = Instant::now();
        let h = s
            .search_code(&format!("unique{}", (k * 7919) % n), 10, None)
            .await
            .unwrap();
        d.push(t.elapsed());
        assert_eq!(h.len(), 1);
    }
    quantiles("recherche_terme_unique", d);

    let mut d = Vec::with_capacity(runs);
    for k in 0..runs {
        let slug = format!("projet-{}", k % PROJETS);
        let t = Instant::now();
        let h = s
            .search_code_in_project("groupe7", 10, None, Some(&slug))
            .await
            .unwrap();
        d.push(t.elapsed());
        std::hint::black_box(h);
    }
    quantiles("recherche_filtre_projet", d);

    let mut d = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let h = s.search_code("*", 10, None).await.unwrap();
        d.push(t.elapsed());
        assert_eq!(h.len(), 10);
    }
    quantiles("placeholder_sans_filtre", d);

    let mut d = Vec::with_capacity(runs);
    for k in 0..runs {
        let slug = format!("projet-{}", k % PROJETS);
        let t = Instant::now();
        let h = s
            .search_code_in_project("*", 10, None, Some(&slug))
            .await
            .unwrap();
        d.push(t.elapsed());
        assert_eq!(h.len(), 10);
    }
    quantiles("placeholder_filtre_projet", d);

    let mut d = Vec::with_capacity(runs);
    for k in 0..runs {
        let path = format!("src/module_{:07}.rs", (k * 104_729) % n);
        let t = Instant::now();
        s.delete_code(&path).await.unwrap();
        d.push(t.elapsed());
    }
    quantiles("delete_code_par_id", d);

    println!("charge_fin\t{}\t-", charge());
    g.close().await.unwrap();
}
