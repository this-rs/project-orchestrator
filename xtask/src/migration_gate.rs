//! `cargo xtask migration-gate` — the ratchet that keeps the Neo4j → obrain
//! migration from regressing while it is still in progress.
//!
//! Two invariants, both checked from the source tree, without a database:
//!
//! 1. **Ported traits never go back.** A store trait is "ported" when a
//!    hand-written `impl <Trait> for Obrain…` exists under `src/obrain/`. The
//!    list in `xtask/migration-portees.txt` records every trait that was ported
//!    when the gate was last reviewed. A trait listed there that is no longer
//!    ported is a REGRESSION (someone deleted a port); a trait ported since is
//!    reported so it can be recorded. The list can only grow.
//!
//! 2. **The default backend stays Neo4j.** The production default is
//!    `DEFAULT_BACKEND = "default"`. Switching the default to obrain before the
//!    migration is finished is refused: the directive of 20/11/2026 is « on ne
//!    branche pas en prod tant qu'on n'a pas migré vers obrain et this-rs ».
//!
//! What is NOT checked here, on purpose: behaviour (the conformance suite
//! `tests/store_conformance.rs` runs every store against Neo4j, the mock and
//! obrain), performance (the bench barrier of `CONSOLIDATION.md`), and data
//! parity on a copy. Those are listed in `docs/migration/GATES.md` with their
//! command and their current status.
//!
//! Modes: by default it prints the report and exits 0 (informational).
//! `--strict` exits 1 on a regression or a changed default. Exit 2 = could not
//! verify (not in a repository, unreadable files).

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::git::Repo;

const HELP: &str = "\
usage: cargo xtask migration-gate [--strict] [--repo <chemin>]

  --strict   exit 1 on a regression (a ported trait lost its port) or when the
             default backend is no longer \"default\"
  --repo     start the repository discovery from this path (default: .)

exit 0 nothing to report (informational), 1 a regression (--strict),
2 could not verify.";

/// The contract files: each `pub trait X` there is a store the backend must serve.
const CONTRACT_FILES: &[&str] = &["src/neo4j/traits.rs", "src/meilisearch/traits.rs"];
/// Where hand-written obrain impls live.
const OBRAIN_DIR: &str = "src/obrain";
/// The ratchet: traits ported when this gate was reviewed. Only grows.
const RATCHET_FILE: &str = "xtask/migration-portees.txt";
/// The storage module (a file before the storage split, a directory after).
const STORAGE_FILES: &[&str] = &["src/storage/mod.rs", "src/storage.rs"];
const DEFAULT_LINE: &str = "pub const DEFAULT_BACKEND: &str = \"default\";";

#[derive(Debug, PartialEq, Eq)]
pub struct Report {
    /// Listed in the ratchet but no longer ported: a regression.
    pub regressions: Vec<String>,
    /// In the contract, not ported, not listed: work still to do.
    pub unported: Vec<String>,
    /// Ported, not yet listed in the ratchet: record them.
    pub newly_ported: Vec<String>,
}

/// Every `pub trait X` declared in a contract file.
pub fn contract_traits(content: &str) -> BTreeSet<String> {
    content
        .lines()
        .filter_map(|l| {
            let rest = l.trim_start().strip_prefix("pub trait ")?;
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

/// Traits that have a hand-written impl for an `Obrain…` type, e.g.
/// `impl crate::neo4j::traits::ProjectStore for ObrainGraphStore {`.
pub fn ported_traits(content: &str) -> BTreeSet<String> {
    content
        .lines()
        .filter_map(|l| {
            let t = l.trim_start();
            // `impl X for …` and `impl<T> X<T> for …` both count.
            let rest = t.strip_prefix("impl")?;
            let rest = rest.strip_prefix('<').map_or(rest, |r| {
                let mut depth = 1usize;
                let end = r
                    .char_indices()
                    .find_map(|(i, c)| {
                        match c {
                            '<' => depth += 1,
                            '>' => depth -= 1,
                            _ => {}
                        }
                        (depth == 0).then_some(i)
                    })
                    .map_or(r.len(), |i| i + 1);
                &r[end..]
            });
            let rest = rest.strip_prefix(' ')?.trim_start();
            let (head, target) = rest.split_once(" for ")?;
            let target = target.trim_start_matches("crate::obrain::");
            if !target.starts_with("Obrain") {
                return None;
            }
            // Drop generic arguments and any module path: keep the trait name.
            let head = head.split('<').next()?.trim();
            let name = head.rsplit("::").next()?.trim();
            (!name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'))
                .then(|| name.to_string())
        })
        .collect()
}

/// The ratchet file: one trait per line, `#` starts a comment.
pub fn ratchet(content: &str) -> BTreeSet<String> {
    content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Compare the contract, the ported set and the ratchet.
pub fn evaluate(
    contract: &BTreeSet<String>,
    ported: &BTreeSet<String>,
    ratchet: &BTreeSet<String>,
) -> Report {
    // Only contract traits count as ports: `impl Debug for ObrainGraphStore`
    // is an impl, not a store port.
    let ported: BTreeSet<String> = ported.intersection(contract).cloned().collect();
    Report {
        regressions: ratchet.difference(&ported).cloned().collect(),
        unported: contract
            .difference(&ported)
            .filter(|t| !ratchet.contains(*t))
            .cloned()
            .collect(),
        newly_ported: ported.difference(ratchet).cloned().collect(),
    }
}

/// True when the storage module still declares the Neo4j default.
pub fn default_is_neo4j(content: &str) -> bool {
    content.lines().any(|l| l.trim() == DEFAULT_LINE)
}

#[derive(Default)]
struct Options {
    strict: bool,
    help: bool,
    repo: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--strict" => o.strict = true,
            "--help" | "-h" => o.help = true,
            "--repo" => o.repo = Some(PathBuf::from(it.next().ok_or("--repo attend un chemin")?)),
            other => return Err(format!("option inconnue « {other} »")),
        }
    }
    Ok(o)
}

/// The only function of this module that touches disk.
pub fn run(args: &[String]) -> i32 {
    let o = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("migration-gate : {e}\n\n{HELP}");
            return 2;
        }
    };
    if o.help {
        println!("{HELP}");
        return 0;
    }
    let start = o.repo.clone().unwrap_or_else(|| PathBuf::from("."));
    let Some(root) = Repo::discover(&start).map(|r| r.path) else {
        eprintln!(
            "migration-gate : non vérifiable — {} n'est pas dans un dépôt git",
            start.display()
        );
        return 2;
    };

    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).ok();

    let mut contract = BTreeSet::new();
    for f in CONTRACT_FILES {
        match read(f) {
            Some(c) => contract.extend(contract_traits(&c)),
            None => {
                eprintln!("migration-gate : non vérifiable — {f} illisible");
                return 2;
            }
        }
    }

    let mut ported = BTreeSet::new();
    let obrain = root.join(OBRAIN_DIR);
    let Ok(entries) = std::fs::read_dir(&obrain) else {
        eprintln!("migration-gate : non vérifiable — {OBRAIN_DIR} illisible");
        return 2;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "rs") {
            if let Ok(c) = std::fs::read_to_string(&p) {
                ported.extend(ported_traits(&c));
            }
        }
    }

    let Some(ratchet_text) = read(RATCHET_FILE) else {
        eprintln!("migration-gate : non vérifiable — {RATCHET_FILE} absent");
        return 2;
    };
    let ratchet_set = ratchet(&ratchet_text);

    let report = evaluate(&contract, &ported, &ratchet_set);

    let storage = STORAGE_FILES.iter().find_map(|f| read(f));
    let default_ok = storage.as_deref().is_some_and(default_is_neo4j);

    // Count only contract traits, as `evaluate` does: an unrelated `impl`
    // (e.g. `Debug`) is not a store port.
    let ported_contract = ported.intersection(&contract).count();
    println!(
        "migration-gate : {} trait(s) de contrat, {} porté(s) à la main, {} dans le cliquet",
        contract.len(),
        ported_contract,
        ratchet_set.len()
    );
    for t in &report.regressions {
        println!("  RÉGRESSION   {t} : listé dans {RATCHET_FILE} mais plus porté");
    }
    for t in &report.newly_ported {
        println!("  à consigner {t} : porté, à ajouter dans {RATCHET_FILE}");
    }
    for t in &report.unported {
        println!("  à porter     {t}");
    }
    if default_ok {
        println!("  backend par défaut : neo4j (« default ») — conforme");
    } else {
        println!("  BACKEND PAR DÉFAUT : n'est plus « default » (ou illisible) — refusé");
    }

    let failed = !report.regressions.is_empty() || !default_ok;
    if failed && o.strict {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(xs: &[&str]) -> BTreeSet<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn contract_traits_are_the_pub_traits_of_the_contract_files() {
        let src = "pub trait ProjectStore: Send {\n    fn x();\n}\n\
                   pub(crate) trait Hidden {}\n  pub trait NoteStore<T> {}\n";
        assert_eq!(contract_traits(src), set(&["NoteStore", "ProjectStore"]));
    }

    #[test]
    fn a_hand_impl_for_an_obrain_type_is_a_port() {
        let src = "impl ProjectStore for ObrainGraphStore {\n\
                   impl crate::neo4j::traits::PlanStore for ObrainGraphStore {\n\
                   impl<T> NoteStore<T> for crate::obrain::ObrainGraphStore {\n\
                   impl ProjectStore for crate::neo4j::client::Neo4jStore {\n\
                   impl Debug for ObrainGraphStore {\n";
        // The parser is generic: it reports every trait with an Obrain target.
        assert_eq!(
            ported_traits(src),
            set(&["Debug", "NoteStore", "PlanStore", "ProjectStore"])
        );
    }

    #[test]
    fn a_non_contract_impl_is_not_a_port() {
        let r = evaluate(
            &set(&["NoteStore"]),
            &set(&["Debug", "NoteStore"]),
            &set(&["NoteStore"]),
        );
        assert!(r.regressions.is_empty());
        assert!(r.newly_ported.is_empty());
        assert!(r.unported.is_empty());
    }

    #[test]
    fn a_listed_trait_that_lost_its_port_is_a_regression() {
        let r = evaluate(
            &set(&["PlanStore", "NoteStore", "SearchStore"]),
            &set(&["NoteStore"]),
            &set(&["PlanStore", "NoteStore"]),
        );
        assert_eq!(r.regressions, vec!["PlanStore".to_string()]);
        assert_eq!(r.unported, vec!["SearchStore".to_string()]);
        assert!(r.newly_ported.is_empty());
    }

    #[test]
    fn a_newly_ported_trait_is_reported_for_the_ratchet_not_refused() {
        let r = evaluate(&set(&["A"]), &set(&["A"]), &set(&[]));
        assert!(r.regressions.is_empty());
        assert_eq!(r.newly_ported, vec!["A".to_string()]);
        assert!(r.unported.is_empty());
    }

    #[test]
    fn the_ratchet_ignores_comments_and_blank_lines() {
        assert_eq!(
            ratchet("# en-tête\n\nNoteStore  # porté\nPlanStore\n"),
            set(&["NoteStore", "PlanStore"])
        );
    }

    #[test]
    fn the_default_backend_must_stay_neo4j() {
        assert!(default_is_neo4j(
            "pub const DEFAULT_BACKEND: &str = \"default\";\nfn x() {}"
        ));
        assert!(!default_is_neo4j(
            "pub const DEFAULT_BACKEND: &str = \"obrain\";"
        ));
        assert!(!default_is_neo4j(""));
    }

    #[test]
    fn options_parse_strict_and_reject_unknown_flags() {
        let o = parse_args(&["--strict".to_string()]).unwrap();
        assert!(o.strict);
        assert!(parse_args(&["--nope".to_string()]).is_err());
        assert!(parse_args(&["--repo".to_string()]).is_err());
    }
}
