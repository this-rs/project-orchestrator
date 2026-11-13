//! Cost of the workspace domain on the obrain backend, on a base of N nodes —
//! the numbers behind the claims of `src/obrain/workspace.rs`.
//!
//! ```text
//! cargo run --release --features obrain-backend,mock-store --example obrain_mesures_workspace -- <nodes>
//! ```
//!
//! The base is filled with `<nodes>` nodes of another label (`remplir_pour_mesure`),
//! then one workspace holds 1 000 milestones; the reads and writes are measured on
//! that workspace. A read that grows with N is a defect: compare the `<nodes>` runs.
//!
//! The base lives in a temporary directory this program creates and removes.
//! Output: one `mesure<TAB>valeur<TAB>unité` line per measure.

use chrono::Utc;
use project_orchestrator::neo4j::models::{MilestoneStatus, WorkspaceMilestoneNode, WorkspaceNode};
use project_orchestrator::neo4j::traits::WorkspaceStore;
use project_orchestrator::obrain::{Creation, ObrainGraphStore};
use std::time::{Duration, Instant};
use uuid::Uuid;

fn espace(i: usize) -> WorkspaceNode {
    let id = Uuid::new_v4();
    WorkspaceNode {
        id,
        name: format!("mesure espace {i:05}"),
        slug: format!("mesure-espace-{}", id.simple()),
        description: Some(format!("espace de mesure {i}")),
        created_at: Utc::now(),
        updated_at: None,
        metadata: serde_json::json!({}),
    }
}

fn jalon(workspace_id: Uuid, i: usize) -> WorkspaceMilestoneNode {
    WorkspaceMilestoneNode {
        id: Uuid::new_v4(),
        workspace_id,
        title: format!("jalon {i:05}"),
        description: Some(format!("jalon de mesure {i}")),
        status: if i % 3 == 0 {
            MilestoneStatus::Completed
        } else {
            MilestoneStatus::Open
        },
        target_date: Some(Utc::now() + chrono::Duration::days((i % 97) as i64)),
        closed_at: None,
        created_at: Utc::now(),
        tags: vec!["mesure".into()],
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
    let noeuds: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .expect("<nodes>");
    println!("base\t{noeuds} nœuds (remplissage), workspace de 1 000 jalons, format neuf");

    let dir = tempfile::Builder::new()
        .prefix("po-obrain-mesures-ws-")
        .tempdir()
        .unwrap();
    let base = dir.path().join("base");

    let store = ObrainGraphStore::open(&base, Creation::Permise).unwrap();
    let rempli = store.remplir_pour_mesure(noeuds, noeuds).unwrap();
    dire("remplissage", rempli.as_secs_f64(), "s");

    let ws = espace(0);
    store.create_workspace(&ws).await.unwrap();
    let mut jalons = Vec::new();
    for i in 0..1_000 {
        let j = jalon(ws.id, i);
        store.create_workspace_milestone(&j).await.unwrap();
        jalons.push(j.id);
    }
    store.close().await.unwrap();
    drop(store);

    // Open: the rebuild reads the workspace labels, not the filler nodes.
    let t = Instant::now();
    let store = ObrainGraphStore::open(&base, Creation::Interdite).unwrap();
    dire("ouverture", t.elapsed().as_secs_f64() * 1e3, "ms");
    // Let the fill's dirty data reach the disk before the fsyncs are measured.
    let _ = std::process::Command::new("sync").status();
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Writes, WAL fsync per write. Each 256th write pays the full flush (O(base)).
    let mut mesures = Vec::new();
    let mut crees = Vec::new();
    for i in 0..200 {
        let e = espace(i + 1);
        crees.push((e.id, e.slug.clone()));
        let t = Instant::now();
        store.create_workspace(&e).await.unwrap();
        mesures.push(t.elapsed());
    }
    dire("create_workspace_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("create_workspace_p99", us(pct(mesures, 0.99)), "µs");

    let mut mesures = Vec::new();
    for i in 0..200 {
        let j = jalon(ws.id, 2_000 + i);
        let t = Instant::now();
        store.create_workspace_milestone(&j).await.unwrap();
        mesures.push(t.elapsed());
    }
    dire("create_milestone_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("create_milestone_p99", us(pct(mesures, 0.99)), "µs");

    // A new title each time: an update that writes nothing new would measure nothing.
    let mut mesures = Vec::new();
    for (k, id) in jalons.iter().take(200).enumerate() {
        let t = Instant::now();
        store
            .update_workspace_milestone(*id, Some(format!("renomme {k}")), None, None, None)
            .await
            .unwrap();
        mesures.push(t.elapsed());
    }
    dire("update_milestone_p50", us(pct(mesures.clone(), 0.5)), "µs");
    dire("update_milestone_p99", us(pct(mesures, 0.99)), "µs");

    // Reads, served from the index.
    let rondes = 200_000u32;
    let t = Instant::now();
    for k in 0..rondes {
        std::hint::black_box(store.get_workspace(ws.id).await.unwrap());
        let _ = k;
    }
    dire(
        "get_workspace",
        t.elapsed().as_secs_f64() * 1e9 / rondes as f64,
        "ns",
    );
    let (_, slug) = &crees[7];
    let t = Instant::now();
    for _ in 0..rondes {
        std::hint::black_box(store.get_workspace_by_slug(slug).await.unwrap());
    }
    dire(
        "get_workspace_by_slug",
        t.elapsed().as_secs_f64() * 1e9 / rondes as f64,
        "ns",
    );

    let mut mesures = Vec::new();
    for _ in 0..200 {
        let t = Instant::now();
        std::hint::black_box(store.list_workspace_milestones(ws.id).await.unwrap());
        mesures.push(t.elapsed());
    }
    dire(
        "list_milestones_1000_p50",
        us(pct(mesures.clone(), 0.5)),
        "µs",
    );
    dire("list_milestones_1000_p99", us(pct(mesures, 0.99)), "µs");

    let mut mesures = Vec::new();
    for _ in 0..200 {
        let t = Instant::now();
        std::hint::black_box(
            store
                .list_workspace_milestones_filtered(ws.id, Some("open"), 50, 0)
                .await
                .unwrap(),
        );
        mesures.push(t.elapsed());
    }
    dire(
        "list_milestones_filtered_p50",
        us(pct(mesures.clone(), 0.5)),
        "µs",
    );
    dire("list_milestones_filtered_p99", us(pct(mesures, 0.99)), "µs");

    store.close().await.unwrap();
}
