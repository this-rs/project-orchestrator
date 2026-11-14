//! `cargo xtask path-pins` — the pinned state of the sibling checkouts.
//!
//! `cargo xtask pins` reads the `rev` pins of Cargo.toml (git dependencies).
//! This one covers the OTHER kind: `path = "../../…"` dependencies, which
//! carry no sha at all. The neighbour can move under our feet and no diff
//! shows it — so the state we build against is written down in `pins.toml`
//! and CONSTATED here. Never writes, never checks out, never merges.

use crate::git::{short, Repo};
use std::path::{Path, PathBuf};

const HELP: &str = "\
usage: cargo xtask path-pins [options]

  --pins <file>   pin file, default pins.toml at the repo root
  --repo <path>   checkout to inspect, default the one containing the cwd
  --fetch         `git fetch` the neighbours first (off by default: an ssh
                  remote without a network would block the gate)

constates two things:
  1. every repo pinned in the pin file is at its pinned commit;
  2. every `path = \"..…\"` dependency of a Cargo.toml is pinned there.

exit 0  every pin holds and every path dependency is pinned
exit 1  a divergence was constated (a neighbour moved, or an unpinned path dep)
exit 2  could not verify (missing clone, unreadable pin file, bad arguments)";

/// One `[repo.NAME]` section of the pin file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub name: String,
    pub path: String,
    pub rev: String,
}

/// Parse the pin file. A section without `rev` is an error: a pin that pins
/// nothing would read as "verified" and is worse than no pin at all.
pub fn parse_pins(text: &str) -> Result<Vec<Pin>, String> {
    let mut out: Vec<Pin> = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line
            .strip_prefix("[repo.")
            .and_then(|r| r.strip_suffix(']'))
        {
            out.push(Pin {
                name: rest.trim().to_string(),
                path: String::new(),
                rev: String::new(),
            });
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"').to_string();
        match key.trim() {
            "path" => {
                let last = out.last_mut().ok_or_else(|| {
                    format!("ligne {} : `path` hors d'une section [repo.<nom>]", n + 1)
                })?;
                last.path = value;
            }
            "rev" => {
                let last = out.last_mut().ok_or_else(|| {
                    format!("ligne {} : `rev` hors d'une section [repo.<nom>]", n + 1)
                })?;
                last.rev = value;
            }
            _ => {}
        }
    }
    for p in &out {
        if p.rev.is_empty() {
            return Err(format!("[repo.{}] n'a pas de `rev`", p.name));
        }
        if p.path.is_empty() {
            return Err(format!("[repo.{}] n'a pas de `path`", p.name));
        }
    }
    Ok(out)
}

/// Every `path = "..…"` value of a manifest — only those that leave the repo.
/// A `path = "crates/x"` is internal and needs no pin.
pub fn path_deps(manifest: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in manifest.lines() {
        let line: Vec<char> = raw.split('#').next().unwrap_or("").chars().collect();
        let mut i = 0;
        while i + 4 <= line.len() {
            let word: String = line[i..i + 4].iter().collect();
            let boundary = i == 0 || !(line[i - 1].is_alphanumeric() || line[i - 1] == '_');
            let mut j = i + 4;
            while j < line.len() && line[j].is_whitespace() {
                j += 1;
            }
            if word == "path" && boundary && j < line.len() && line[j] == '=' {
                j += 1;
                while j < line.len() && line[j].is_whitespace() {
                    j += 1;
                }
                if j < line.len() && (line[j] == '"' || line[j] == '\'') {
                    let quote = line[j];
                    j += 1;
                    let start = j;
                    while j < line.len() && line[j] != quote {
                        j += 1;
                    }
                    let value: String = line[start..j].iter().collect();
                    if value.starts_with("..") {
                        out.push(value);
                    }
                }
            }
            i += 1;
        }
    }
    out
}

/// Resolve a pin path against the repo root, then each of its ANCESTORS.
///
/// The neighbours do not live at the same depth from every checkout (the worktree
/// is `worktrees/obrain-default`, the main checkout is `backend`, and both sit
/// under the same `projects/` tree). A fixed base would be wrong for one of them,
/// so the first ancestor where the path exists wins — deterministic, and no
/// absolute path in a versioned file.
fn resolve_pin(root: &Path, rel: &str) -> Option<PathBuf> {
    let mut base: Option<PathBuf> = Some(root.to_path_buf());
    while let Some(b) = base {
        let candidate = b.join(rel);
        if candidate.is_dir() {
            return Some(candidate);
        }
        base = b.parent().map(Path::to_path_buf);
    }
    None
}

fn manifests(root: &Path) -> Vec<PathBuf> {
    let mut out = vec![root.join("Cargo.toml")];
    if let Ok(entries) = std::fs::read_dir(root.join("crates")) {
        for e in entries.flatten() {
            let m = e.path().join("Cargo.toml");
            if m.is_file() {
                out.push(m);
            }
        }
    }
    let desktop = root.join("desktop").join("src-tauri").join("Cargo.toml");
    if desktop.is_file() {
        out.push(desktop);
    }
    out.retain(|p| p.is_file());
    out
}

fn value(args: &[String], i: &mut usize, flag: &str) -> Result<String, i32> {
    *i += 1;
    match args.get(*i) {
        Some(v) => Ok(v.clone()),
        None => {
            eprintln!("xtask path-pins: {flag} attend une valeur");
            Err(2)
        }
    }
}

pub fn run(args: &[String]) -> i32 {
    let mut pins_file: Option<PathBuf> = None;
    let mut repo_path: Option<PathBuf> = None;
    // Fetch is OPT-IN: a sibling checkout may track an ssh remote that hangs
    // without a network. A gate that blocks is worse than a gate that judges
    // the refs it already has.
    let mut fetch = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => {
                println!("{HELP}");
                return 0;
            }
            "--fetch" => fetch = true,
            "--no-fetch" => fetch = false,
            "--pins" => match value(args, &mut i, "--pins") {
                Ok(v) => pins_file = Some(PathBuf::from(v)),
                Err(c) => return c,
            },
            "--repo" => match value(args, &mut i, "--repo") {
                Ok(v) => repo_path = Some(PathBuf::from(v)),
                Err(c) => return c,
            },
            other => {
                eprintln!("xtask path-pins: option inconnue « {other} »\n\n{HELP}");
                return 2;
            }
        }
        i += 1;
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let root = match repo_path {
        Some(p) => p,
        None => match Repo::discover(&cwd) {
            Some(r) => r.path,
            None => {
                eprintln!(
                    "xtask path-pins: aucun dépôt git au-dessus de {}",
                    cwd.display()
                );
                return 2;
            }
        },
    };

    let pins_path = pins_file.unwrap_or_else(|| root.join("pins.toml"));
    let text = match std::fs::read_to_string(&pins_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("xtask path-pins: {} illisible : {e}", pins_path.display());
            return 2;
        }
    };
    let pins = match parse_pins(&text) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("xtask path-pins: {} invalide : {e}", pins_path.display());
            return 2;
        }
    };

    let mut divergences: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut absent: Vec<String> = Vec::new();

    // 1. every pinned repo is at its pinned commit.
    for pin in &pins {
        let Some(dir) = resolve_pin(&root, &pin.path) else {
            absent.push(pin.name.clone());
            continue;
        };
        let Some(repo) = Repo::discover(&dir) else {
            absent.push(pin.name.clone());
            continue;
        };
        if fetch {
            // Best effort: a network-less machine keeps the refs it has.
            let _ = repo.fetch("origin");
        }
        let Some(head) = repo.resolve("HEAD") else {
            eprintln!(
                "xtask path-pins: [repo.{}] HEAD illisible, non vérifiable",
                pin.name
            );
            return 2;
        };
        checked += 1;
        if head != pin.rev {
            let mut how = String::from("a bougé");
            if let Ok(n) = repo.count_between(&pin.rev, &head) {
                let other = repo.count_between(&head, &pin.rev).unwrap_or(0);
                how = format!("{n} commit(s) d'avance, {other} de retard");
            }
            divergences.push(format!(
                "[repo.{}] épinglé {} mais à {} ({how}) — {}",
                pin.name,
                short(&pin.rev),
                short(&head),
                dir.display()
            ));
        }
    }

    // 2. every out-of-repo `path = "..…"` dependency is pinned.
    let mut unpinned = 0usize;
    for manifest in manifests(&root) {
        let Ok(content) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        let base = manifest.parent().unwrap_or(&root).to_path_buf();
        for dep in path_deps(&content) {
            let target = base.join(&dep);
            let Some(repo) = Repo::discover(&target) else {
                divergences.push(format!(
                    "`path = \"{dep}\"` dans {} : aucun dépôt git à {}",
                    manifest.display(),
                    target.display()
                ));
                unpinned += 1;
                continue;
            };
            // `../neural-routing-core` or `../..`: a path that climbs but stays
            // inside THIS repository is not a neighbour and needs no pin. The
            // toplevel decides, not the shape of the string.
            let same_repo =
                std::fs::canonicalize(&repo.path).ok() == std::fs::canonicalize(&root).ok();
            if same_repo {
                continue;
            }
            let target_repo = std::fs::canonicalize(&repo.path).ok();
            let pinned = pins.iter().any(|p| {
                resolve_pin(&root, &p.path).and_then(|d| std::fs::canonicalize(d).ok())
                    == target_repo
            });
            if !pinned {
                divergences.push(format!(
                    "`path = \"{dep}\"` dans {} : {} n'est PAS épinglé dans {}",
                    manifest.display(),
                    repo.path.display(),
                    pins_path.display()
                ));
                unpinned += 1;
            }
        }
    }

    if divergences.is_empty() {
        println!(
            "path-pins: {checked} dépôt(s) épinglé(s) conforme(s), aucune dépendance par chemin non épinglée"
        );
        if !absent.is_empty() {
            println!(
                "  absent(s) du disque, et ignoré(s) — aucune dépendance ne les référence : {}",
                absent.join(", ")
            );
        }
        return 0;
    }

    println!("path-pins: ÉCART CONSTATÉ");
    for d in &divergences {
        println!("  {d}");
    }
    if unpinned > 0 {
        println!();
        println!("Une dépendance `path` non épinglée ne dit aucun commit : le voisin peut");
        println!(
            "bouger sans qu'aucun diff ne le montre. Ajouter sa section dans {} :",
            pins_path.display()
        );
        println!("  [repo.<nom>]");
        println!("  path = \"../<chemin>\"");
        println!("  rev  = \"<sha de 40 caractères>\"");
    }
    println!();
    println!("Si le déplacement est voulu : mettre à jour le `rev` dans le même commit.");
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINS: &str = r#"
version = 1

[repo.this]
path = "../../this-rs/this"
url = "https://github.com/this-rs/this.git"
rev = "3978baf8e2c9046c4c7ed57ceeb2f9824c84c2f4"   # comment

[repo.obrain]
path = "../../obrain/obrain-db"
rev = "36082d87a70a665325a6ee66450db13e20eb08ff"
"#;

    #[test]
    fn parse_pins_reads_sections_paths_and_revs() {
        let pins = parse_pins(PINS).expect("valid");
        assert_eq!(pins.len(), 2);
        assert_eq!(pins[0].name, "this");
        assert_eq!(pins[0].path, "../../this-rs/this");
        assert_eq!(pins[0].rev.len(), 40);
        assert_eq!(pins[1].name, "obrain");
    }

    #[test]
    fn a_pin_without_rev_is_an_error_not_a_pass() {
        let bad = "[repo.x]\npath = \"../x\"\n";
        assert!(
            parse_pins(bad).is_err(),
            "un pin sans rev ne doit jamais lire comme vérifié"
        );
    }

    #[test]
    fn path_deps_keeps_only_the_ones_leaving_the_repo() {
        let manifest = r#"
[dependencies]
internal = { path = "crates/internal" }
this = { path = "../this-rs/this" }
obrain = { path = '../../obrain/obrain-db/crates/obrain' }
serde = "1"
"#;
        assert_eq!(
            path_deps(manifest),
            vec![
                "../this-rs/this".to_string(),
                "../../obrain/obrain-db/crates/obrain".to_string()
            ]
        );
    }

    #[test]
    fn comments_and_inline_comments_never_create_a_pin() {
        let text = "# [repo.fake]\nversion = 1\n[repo.real] # the real one\npath = \"../real\"\nrev = \"abc1234\"\n";
        let pins = parse_pins(text).expect("valid");
        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].name, "real");
    }
}
