//! MEM-1 — mesure reproductible de l'empreinte mémoire de l'embedder local.
//!
//! Charge le modèle fastembed comme `FastEmbedProvider` le fait, découpe un
//! gros fichier source en morceaux de `--chunk` caractères (chacun tronqué à
//! 512 jetons par le tokenizer : c'est le pire cas du padding `BatchLongest`),
//! puis embarque le corpus `--passes` fois avec la taille de lot demandée, en
//! relevant l'empreinte physique (pas le RSS) avant et après chaque étape.
//!
//! `--batch 0` = `None` (défaut fastembed : 256), ce que le backend fait
//! aujourd'hui. Comparer avec `--batch 16`.
//!
//! ```text
//! cargo run --release --example mem_embed -- --batch 0  --passes 2
//! cargo run --release --example mem_embed -- --batch 16 --passes 2
//! ```
//!
//! Sur macOS la mesure passe par `footprint -p <pid>` (le même outil que celui
//! du diagnostic) ; sur Linux par `/proc/self/status` (VmRSS + VmSwap, VmHWM).
//! Pas d'autre dépendance que celles du crate.

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::time::Instant;

struct Footprint {
    current_mb: f64,
    peak_mb: f64,
}

fn to_mb(value: f64, unit: &str) -> f64 {
    match unit.to_ascii_uppercase().as_str() {
        "KB" => value / 1024.0,
        "MB" => value,
        "GB" => value * 1024.0,
        _ => value,
    }
}

#[cfg(target_os = "macos")]
fn footprint() -> Option<Footprint> {
    let out = std::process::Command::new("footprint")
        .arg("-p")
        .arg(std::process::id().to_string())
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut current = None;
    let mut peak = None;
    for line in text.lines() {
        let line = line.trim();
        let parse = |rest: &str| -> Option<f64> {
            let mut it = rest.split_whitespace();
            let v: f64 = it.next()?.replace(',', ".").parse().ok()?;
            let u = it.next().unwrap_or("MB");
            Some(to_mb(v, u))
        };
        if let Some(rest) = line.strip_prefix("phys_footprint_peak:") {
            peak = parse(rest.trim());
        } else if let Some(rest) = line.strip_prefix("phys_footprint:") {
            current = parse(rest.trim());
        }
    }
    Some(Footprint {
        current_mb: current?,
        peak_mb: peak.unwrap_or(0.0),
    })
}

#[cfg(target_os = "linux")]
fn footprint() -> Option<Footprint> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss = 0.0;
    let mut swap = 0.0;
    let mut hwm = 0.0;
    for line in status.lines() {
        let kb = |l: &str| -> f64 {
            l.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
                / 1024.0
        };
        if line.starts_with("VmRSS:") {
            rss = kb(line);
        } else if line.starts_with("VmSwap:") {
            swap = kb(line);
        } else if line.starts_with("VmHWM:") {
            hwm = kb(line);
        }
    }
    Some(Footprint {
        current_mb: rss + swap,
        peak_mb: hwm,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn footprint() -> Option<Footprint> {
    None
}

fn report(label: &str, started: Option<Instant>) {
    let elapsed = started
        .map(|s| format!(" ({:.1} s)", s.elapsed().as_secs_f64()))
        .unwrap_or_default();
    match footprint() {
        Some(f) => println!(
            "{label:<28} footprint = {:>8.0} MB   peak = {:>8.0} MB{elapsed}",
            f.current_mb, f.peak_mb
        ),
        None => println!("{label:<28} footprint indisponible sur cette plateforme{elapsed}"),
    }
}

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next();
        }
    }
    None
}

fn chunk_text(text: &str, chunk_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::with_capacity(chunk_chars);
    for ch in text.chars() {
        current.push(ch);
        if current.chars().count() >= chunk_chars {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

fn main() -> anyhow::Result<()> {
    let batch: usize = arg("--batch").map(|v| v.parse()).transpose()?.unwrap_or(0);
    let passes: usize = arg("--passes").map(|v| v.parse()).transpose()?.unwrap_or(2);
    let chunk: usize = arg("--chunk")
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2000);
    let file = arg("--file").unwrap_or_else(|| "src/chat/manager.rs".to_string());
    let batch_opt = if batch == 0 { None } else { Some(batch) };

    let text = std::fs::read_to_string(&file)?;
    let texts = chunk_text(&text, chunk);
    println!(
        "corpus: {file} -> {} morceaux de ~{chunk} caractères ; lot = {}",
        texts.len(),
        batch_opt.map_or("None (défaut fastembed 256)".to_string(), |b| b
            .to_string())
    );

    report("avant chargement", None);

    let cache_dir: PathBuf = std::env::var("EMBEDDING_FASTEMBED_CACHE_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".fastembed_cache")))
        .expect("home dir");
    let t = Instant::now();
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::MultilingualE5Base)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(false),
    )?;
    report("après chargement modèle", Some(t));

    for pass in 1..=passes {
        let t = Instant::now();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let vectors = model.embed(refs, batch_opt)?;
        assert_eq!(vectors.len(), texts.len());
        report(&format!("après passe {pass}"), Some(t));
    }

    drop(model);
    report("après drop du modèle", None);
    Ok(())
}
