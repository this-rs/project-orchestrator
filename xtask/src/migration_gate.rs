//! `cargo xtask migration-gate` — the ratchet that keeps the Neo4j → obrain
//! migration from regressing while it is still in progress.
//!
//! Checked from the source tree, without a database:
//!
//! 1. **Ported traits never go back (G2).** A store trait is "ported" when a
//!    hand-written `impl <Trait> for Obrain…` exists anywhere under
//!    `src/obrain/` (recursively). Comments and string literals are removed
//!    before the search, so an impl left in a comment or a string is not a
//!    port. Only the traits of the contract (`pub trait` of
//!    `src/neo4j/traits.rs` and `src/meilisearch/traits.rs`) count.
//!    `xtask/migration-portees.txt` lists the ported traits; in `--strict`:
//!    - a listed trait that is no longer ported is a REGRESSION;
//!    - a ported trait that is not listed must be recorded in the same change;
//!    - with `--base <rev>`, a line removed from the list since `<rev>` is
//!      refused: the list can only grow, and that is enforced, not just written.
//!
//! 2. **The default backend stays Neo4j (G3).** Directive of 20/11/2026: « on
//!    ne branche pas en prod tant qu'on n'a pas migré vers obrain et this-rs ».
//!    Refused when: `DEFAULT_BACKEND` is not defined exactly once with the
//!    value `"default"`; the configuration does not fall back to it
//!    (`src/lib.rs`); a tracked deployment file sets `STORAGE_BACKEND` (or a
//!    YAML `backend:`) to obrain. That the default constant maps to the Neo4j
//!    backend is proved by the unit test
//!    `default_backend_is_selected_for_the_default_value` (`src/storage`).
//!
//! What is NOT checked here: behaviour (`tests/store_conformance.rs`),
//! performance, parity on a copy. See `docs/migration/GATES.md`.
//!
//! Exit codes: 0 nothing to report, or informational mode; 1 a finding
//! (`--strict`); 2 could not verify (not a repository, an unreadable file, an
//! unknown `--base`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::git::Repo;

const HELP: &str = "\
usage: cargo xtask migration-gate [--strict] [--base <rev>] [--repo <chemin>]

  --strict   exit 1 on any finding: a ported trait lost its port, a ported trait
             is not recorded, the ratchet shrank since --base, or the default
             backend is no longer Neo4j
  --base     compare the ratchet with the one at this revision (the target
             branch of a pull request). Empty or all-zero = no comparison.
  --repo     start the repository discovery from this path (default: .)

exit 0 nothing to report (or informational mode), 1 a finding (--strict),
2 could not verify.";

const CONTRACT_FILES: &[&str] = &["src/neo4j/traits.rs", "src/meilisearch/traits.rs"];
const OBRAIN_DIR: &str = "src/obrain";
const RATCHET_FILE: &str = "xtask/migration-portees.txt";
/// The storage module (a file before the storage split, a directory after).
const STORAGE_FILES: &[&str] = &["src/storage/mod.rs", "src/storage.rs"];
const CONFIG_FILE: &str = "src/lib.rs";
/// How the configuration falls back to the default backend.
const DEFAULT_WIRING: &str = "unwrap_or_else(|| crate::storage::DEFAULT_BACKEND.to_string())";

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Remove comments (line, nested block) and, when `strings` is false, the
/// contents of string literals (regular, byte, raw). Removed text becomes
/// spaces; newlines are kept. Char literals are skipped so that `'"'` does not
/// open a string.
pub fn strip_code(content: &str, strings: bool) -> String {
    let c: Vec<char> = content.chars().collect();
    let mut out = String::with_capacity(content.len());
    let mut i = 0;
    let blank = |ch: char| if ch == '\n' { '\n' } else { ' ' };
    while i < c.len() {
        let ch = c[i];
        let next = c.get(i + 1).copied();
        // Line comment.
        if ch == '/' && next == Some('/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        // Block comment, nested.
        if ch == '/' && next == Some('*') {
            let mut depth = 0usize;
            while i < c.len() {
                if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
            continue;
        }
        // Raw string: r"…", r#"…"#, br"…" — `r` not inside an identifier.
        let prev_ident =
            i > 0 && is_ident(c[i - 1]) && !(c[i - 1] == 'b' && (i < 2 || !is_ident(c[i - 2])));
        if ch == 'r' && !prev_ident {
            let mut j = i + 1;
            while c.get(j) == Some(&'#') {
                j += 1;
            }
            if c.get(j) == Some(&'"') {
                let hashes = j - i - 1;
                // Opening r##" is kept as is.
                out.extend(&c[i..=j]);
                i = j + 1;
                loop {
                    if i >= c.len() {
                        break;
                    }
                    if c[i] == '"' && (0..hashes).all(|h| c.get(i + 1 + h) == Some(&'#')) {
                        out.push('"');
                        for _ in 0..hashes {
                            out.push('#');
                        }
                        i += 1 + hashes;
                        break;
                    }
                    out.push(if strings { c[i] } else { blank(c[i]) });
                    i += 1;
                }
                continue;
            }
        }
        // Regular (or byte) string.
        if ch == '"' {
            out.push('"');
            i += 1;
            while i < c.len() {
                if c[i] == '\\' {
                    let keep = |x: char| if strings { x } else { blank(x) };
                    out.push(keep(c[i]));
                    if let Some(&e) = c.get(i + 1) {
                        out.push(keep(e));
                    }
                    i += 2;
                    continue;
                }
                if c[i] == '"' {
                    out.push('"');
                    i += 1;
                    break;
                }
                out.push(if strings { c[i] } else { blank(c[i]) });
                i += 1;
            }
            continue;
        }
        // Char literal ('x', '\n', '\''), not a lifetime ('a).
        if ch == '\'' {
            if next == Some('\\') {
                // Skip the escaped char first: '\'' ends at i + 3.
                let mut j = i + 3;
                while j < c.len() && c[j] != '\'' {
                    j += 1;
                }
                out.extend(c[i..=j.min(c.len() - 1)].iter().map(|&x| blank(x)));
                i = j + 1;
                continue;
            }
            if c.get(i + 2) == Some(&'\'') {
                out.push_str("   ");
                i += 3;
                continue;
            }
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// Collapse all whitespace runs into single spaces.
fn flatten(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every `pub trait X` declared in a contract file (comments ignored).
pub fn contract_traits(content: &str) -> BTreeSet<String> {
    let flat = flatten(&strip_code(content, false));
    let mut out = BTreeSet::new();
    let mut rest = flat.as_str();
    while let Some(p) = rest.find("pub trait ") {
        let after = &rest[p + "pub trait ".len()..];
        let name: String = after.chars().take_while(|c| is_ident(*c)).collect();
        if !name.is_empty() {
            out.insert(name);
        }
        rest = after;
    }
    out
}

/// Traits that have an `impl <Trait> for Obrain…` (any line layout, comments
/// and strings ignored). The parser is generic: `evaluate` keeps only the
/// contract traits.
pub fn ported_traits(content: &str) -> BTreeSet<String> {
    let flat = flatten(&strip_code(content, false));
    let s = flat;
    let mut out = BTreeSet::new();
    let mut idx = 0;
    while let Some(off) = s[idx..].find("impl") {
        let p = idx + off;
        idx = p + 4;
        let before_ok = p == 0 || !s[..p].chars().last().is_some_and(is_ident);
        let after = &s[p + 4..];
        if !before_ok || !(after.starts_with(' ') || after.starts_with('<')) {
            continue;
        }
        // The header runs to the opening brace (or `;` for a declaration).
        let end = after.find(['{', ';']).unwrap_or(after.len());
        let header = &after[..end];
        // Skip `<…>` generics right after `impl`.
        let header = header.strip_prefix('<').map_or(header, |r| {
            let mut depth = 1usize;
            let cut = r
                .char_indices()
                .find_map(|(i, c)| {
                    match c {
                        '<' => depth += 1,
                        '>' => depth -= 1,
                        _ => {}
                    }
                    (depth == 0).then_some(i + 1)
                })
                .unwrap_or(r.len());
            &r[cut..]
        });
        let Some((trait_part, target)) = header.trim().split_once(" for ") else {
            continue;
        };
        let mut target = target.trim();
        for prefix in ["crate::obrain::", "super::", "self::", "crate::"] {
            target = target.strip_prefix(prefix).unwrap_or(target);
        }
        if !target.starts_with("Obrain") {
            continue;
        }
        let name = trait_part
            .split('<')
            .next()
            .unwrap_or("")
            .trim()
            .rsplit("::")
            .next()
            .unwrap_or("")
            .trim();
        if !name.is_empty() && name.chars().all(is_ident) {
            out.insert(name.to_string());
        }
    }
    out
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

#[derive(Debug, PartialEq, Eq, Default)]
pub struct Report {
    /// Listed in the ratchet but no longer ported.
    pub regressions: Vec<String>,
    /// In the contract, not ported, not listed: work still to do.
    pub unported: Vec<String>,
    /// Ported, not listed: must be recorded in the same change.
    pub newly_ported: Vec<String>,
    /// Listed at `--base`, removed from the ratchet since.
    pub shrunk: Vec<String>,
}

/// Compare the contract, the ported set, the ratchet and (optionally) the
/// ratchet at the base revision.
pub fn evaluate(
    contract: &BTreeSet<String>,
    ported: &BTreeSet<String>,
    ratchet: &BTreeSet<String>,
    base_ratchet: Option<&BTreeSet<String>>,
) -> Report {
    // Only contract traits are ports: `impl Debug for ObrainGraphStore` is not.
    let ported: BTreeSet<String> = ported.intersection(contract).cloned().collect();
    Report {
        regressions: ratchet.difference(&ported).cloned().collect(),
        unported: contract
            .difference(&ported)
            .filter(|t| !ratchet.contains(*t))
            .cloned()
            .collect(),
        newly_ported: ported.difference(ratchet).cloned().collect(),
        shrunk: base_ratchet
            .map(|b| b.difference(ratchet).cloned().collect())
            .unwrap_or_default(),
    }
}

/// Findings on the default backend; empty = conforme.
pub fn default_findings(storage: &str, config: &str) -> Vec<String> {
    let mut f = Vec::new();
    // Comments removed, strings kept: a definition hidden in a string still
    // counts as a second definition, which is refused (fail-closed).
    let s = flatten(&strip_code(storage, true));
    let defs: Vec<&str> = s
        .match_indices("const DEFAULT_BACKEND")
        .map(|(i, _)| &s[i..])
        .collect();
    match defs.as_slice() {
        [] => f.push("DEFAULT_BACKEND n'est plus défini".to_string()),
        [one] => {
            let value = one
                .split_once('=')
                .and_then(|(_, v)| v.split_once(';'))
                .map(|(v, _)| v.trim())
                .unwrap_or("");
            if value != "\"default\"" {
                f.push(format!(
                    "DEFAULT_BACKEND vaut {value} au lieu de \"default\""
                ));
            }
        }
        many => f.push(format!(
            "DEFAULT_BACKEND est défini {} fois (une seule définition, \"default\", est admise)",
            many.len()
        )),
    }
    let cfg = flatten(&strip_code(config, true));
    if !cfg.contains(&flatten(DEFAULT_WIRING)) {
        f.push(format!(
            "{CONFIG_FILE} ne retombe plus sur DEFAULT_BACKEND quand rien n'est configuré"
        ));
    }
    f
}

/// Lines of a deployment file that select obrain as the storage backend.
pub fn deploy_findings(path: &str, content: &str) -> Vec<String> {
    content
        .lines()
        .enumerate()
        .filter_map(|(n, l)| {
            let l = l.split('#').next().unwrap_or("").trim().to_lowercase();
            let env = l.contains("storage_backend") && l.contains("obrain");
            let yaml = l.starts_with("backend:") && l.contains("obrain");
            (env || yaml).then(|| format!("{path}:{} sélectionne obrain", n + 1))
        })
        .collect()
}

/// Is this tracked file a deployment configuration to scan?
fn is_deploy_file(path: &str) -> bool {
    let skip = [
        "docs/",
        "tests/",
        "examples/",
        ".github/",
        "xtask/",
        "src/",
        "build/",
    ];
    if skip.iter().any(|p| path.starts_with(p)) || path.ends_with(".md") {
        return false;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    name.ends_with(".yml")
        || name.ends_with(".yaml")
        || name.ends_with(".env")
        || name.starts_with(".env")
        || name.starts_with("Dockerfile")
        || name.starts_with("docker-compose")
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("{} illisible : {e}", dir.display()))?;
    for e in entries {
        let p = e
            .map_err(|e| format!("{} illisible : {e}", dir.display()))?
            .path();
        if p.is_dir() {
            rs_files(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    Ok(())
}

#[derive(Default)]
struct Options {
    strict: bool,
    help: bool,
    repo: Option<PathBuf>,
    base: Option<String>,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--strict" => o.strict = true,
            "--help" | "-h" => o.help = true,
            "--repo" => o.repo = Some(PathBuf::from(it.next().ok_or("--repo attend un chemin")?)),
            "--base" => o.base = Some(it.next().ok_or("--base attend une révision")?.clone()),
            other => return Err(format!("option inconnue « {other} »")),
        }
    }
    Ok(o)
}

/// An absent base: no comparison (first push of a branch, local run).
fn base_is_absent(rev: &str) -> bool {
    rev.trim().is_empty() || rev.trim().chars().all(|c| c == '0')
}

/// The only function of this module that touches disk or `git`.
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
    let unverifiable = |why: String| {
        eprintln!("migration-gate : non vérifiable — {why}");
        2
    };
    let start = o.repo.clone().unwrap_or_else(|| PathBuf::from("."));
    let Some(repo) = Repo::discover(&start) else {
        return unverifiable(format!("{} n'est pas dans un dépôt git", start.display()));
    };
    let root = repo.path.clone();
    let read = |rel: &str| {
        std::fs::read_to_string(root.join(rel)).map_err(|e| format!("{rel} illisible : {e}"))
    };

    let mut contract = BTreeSet::new();
    for f in CONTRACT_FILES {
        match read(f) {
            Ok(c) => contract.extend(contract_traits(&c)),
            Err(e) => return unverifiable(e),
        }
    }

    let mut files = Vec::new();
    if let Err(e) = rs_files(&root.join(OBRAIN_DIR), &mut files) {
        return unverifiable(e);
    }
    let mut ported = BTreeSet::new();
    for p in &files {
        match std::fs::read_to_string(p) {
            Ok(c) => ported.extend(ported_traits(&c)),
            Err(e) => return unverifiable(format!("{} illisible : {e}", p.display())),
        }
    }

    let ratchet_set = match read(RATCHET_FILE) {
        Ok(t) => ratchet(&t),
        Err(e) => return unverifiable(e),
    };

    let base_set = match o.base.as_deref() {
        None => None,
        Some(rev) if base_is_absent(rev) => None,
        Some(rev) => {
            if repo.resolve(rev).is_none() {
                return unverifiable(format!("révision de base « {rev} » inconnue"));
            }
            // Absent at the base (the ratchet is new there): nothing to compare.
            let spec = format!("{rev}:{RATCHET_FILE}");
            Some(
                repo.git(&["show", &spec])
                    .map(|t| ratchet(&t))
                    .unwrap_or_default(),
            )
        }
    };

    let report = evaluate(&contract, &ported, &ratchet_set, base_set.as_ref());

    let Some(storage) = STORAGE_FILES.iter().find_map(|f| read(f).ok()) else {
        return unverifiable("module de stockage illisible".to_string());
    };
    let config = match read(CONFIG_FILE) {
        Ok(c) => c,
        Err(e) => return unverifiable(e),
    };
    let mut defaults = default_findings(&storage, &config);
    match repo.git(&["ls-files"]) {
        Ok(list) => {
            for path in list.lines().filter(|p| is_deploy_file(p)) {
                if let Ok(c) = read(path) {
                    defaults.extend(deploy_findings(path, &c));
                }
            }
        }
        Err(e) => return unverifiable(format!("git ls-files : {e}")),
    }

    let ported_contract = ported.intersection(&contract).count();
    println!(
        "migration-gate : {} trait(s) de contrat, {} porté(s) à la main, {} dans le cliquet{}",
        contract.len(),
        ported_contract,
        ratchet_set.len(),
        match &base_set {
            Some(b) => format!(", {} dans le cliquet de base", b.len()),
            None => String::new(),
        }
    );
    for t in &report.regressions {
        println!("  RÉGRESSION       {t} : listé dans {RATCHET_FILE} mais plus porté");
    }
    for t in &report.shrunk {
        println!("  CLIQUET RÉDUIT   {t} : retiré de {RATCHET_FILE} depuis la base");
    }
    for t in &report.newly_ported {
        println!(
            "  NON CONSIGNÉ     {t} : porté, à ajouter dans {RATCHET_FILE} dans le même changement"
        );
    }
    for t in &report.unported {
        println!("  à porter         {t}");
    }
    if defaults.is_empty() {
        println!(
            "  backend par défaut : neo4j (« default »), câblé dans la configuration — conforme"
        );
    } else {
        for d in &defaults {
            println!("  BACKEND PAR DÉFAUT : {d} — refusé");
        }
    }

    let failed = !report.regressions.is_empty()
        || !report.shrunk.is_empty()
        || !report.newly_ported.is_empty()
        || !defaults.is_empty();
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

    const WIRED: &str = "x.unwrap_or_else(|| crate::storage::DEFAULT_BACKEND.to_string())";

    #[test]
    fn comments_and_strings_are_removed_but_lines_kept() {
        let src = "a // impl X for ObrainA {\nb /* impl Y /* nested */ for ObrainB { */ c\n\
                   let s = \"impl Z for ObrainC {\"; let r = r#\"impl W for ObrainD {\"#;\n\
                   let q = '\"'; let e = '\\''; let l: &'a str = x;";
        let out = strip_code(src, false);
        assert_eq!(out.lines().count(), src.lines().count());
        assert!(!out.contains("impl"), "{out}");
        assert!(
            out.contains("&'a str"),
            "a lifetime is not a char literal: {out}"
        );
        let kept = strip_code("let s = \"keep\"; // gone", true);
        assert!(kept.contains("\"keep\"") && !kept.contains("gone"));
    }

    #[test]
    fn contract_traits_are_the_pub_traits_of_the_contract_files() {
        let src = "pub trait ProjectStore: Send {\n    fn x();\n}\n\
                   pub(crate) trait Hidden {}\n  pub trait NoteStore<T> {}\n// pub trait Ghost {}\n";
        assert_eq!(contract_traits(src), set(&["NoteStore", "ProjectStore"]));
    }

    #[test]
    fn a_hand_impl_for_an_obrain_type_is_a_port_in_any_layout() {
        let src = "impl ProjectStore for ObrainGraphStore {\n\
                   impl crate::neo4j::traits::PlanStore\n    for crate::obrain::ObrainGraphStore\n{\n\
                   impl<T> NoteStore<T> for super::ObrainGraphStore {\n\
                   impl ChatStore for self::ObrainGraphStore {\n\
                   impl ProjectStore for crate::neo4j::client::Neo4jStore {\n\
                   impl Debug for ObrainGraphStore {\n";
        assert_eq!(
            ported_traits(src),
            set(&[
                "ChatStore",
                "Debug",
                "NoteStore",
                "PlanStore",
                "ProjectStore"
            ])
        );
    }

    #[test]
    fn an_impl_in_a_comment_or_a_string_is_not_a_port() {
        let src = "/* impl ProjectStore for ObrainGraphStore { */\n\
                   // impl PlanStore for ObrainGraphStore {\n\
                   const S: &str = \"impl NoteStore for ObrainGraphStore {\";\n\
                   const R: &str = r#\"impl ChatStore for ObrainGraphStore {\"#;\n";
        assert!(ported_traits(src).is_empty(), "{:?}", ported_traits(src));
    }

    #[test]
    fn a_listed_trait_that_lost_its_port_is_a_regression() {
        let r = evaluate(
            &set(&["PlanStore", "NoteStore", "SearchStore"]),
            &set(&["NoteStore"]),
            &set(&["PlanStore", "NoteStore"]),
            None,
        );
        assert_eq!(r.regressions, vec!["PlanStore".to_string()]);
        assert_eq!(r.unported, vec!["SearchStore".to_string()]);
        assert!(r.newly_ported.is_empty() && r.shrunk.is_empty());
    }

    #[test]
    fn a_ported_trait_not_recorded_is_a_finding() {
        let r = evaluate(&set(&["A"]), &set(&["A"]), &set(&[]), None);
        assert!(r.regressions.is_empty());
        assert_eq!(r.newly_ported, vec!["A".to_string()]);
    }

    #[test]
    fn removing_a_port_and_its_ratchet_line_together_is_caught_by_the_base() {
        // The trick the review found: delete the port AND the line.
        let r = evaluate(
            &set(&["ProjectStore", "NoteStore"]),
            &set(&["NoteStore"]),
            &set(&["NoteStore"]),
            Some(&set(&["ProjectStore", "NoteStore"])),
        );
        assert!(r.regressions.is_empty());
        assert_eq!(r.shrunk, vec!["ProjectStore".to_string()]);
    }

    #[test]
    fn a_non_contract_impl_is_not_a_port() {
        let r = evaluate(
            &set(&["NoteStore"]),
            &set(&["Debug", "NoteStore"]),
            &set(&["NoteStore"]),
            None,
        );
        assert_eq!(r, Report::default());
    }

    #[test]
    fn the_ratchet_ignores_comments_and_blank_lines() {
        assert_eq!(
            ratchet("# en-tête\n\nNoteStore  # porté\nPlanStore\n"),
            set(&["NoteStore", "PlanStore"])
        );
    }

    #[test]
    fn the_default_backend_is_checked_by_its_definition_and_its_wiring() {
        let ok = "pub const DEFAULT_BACKEND: &str = \"default\"; // neo4j";
        assert!(default_findings(ok, WIRED).is_empty());
        let wrapped = "pub const DEFAULT_BACKEND: &str =\n    \"default\";";
        assert!(default_findings(wrapped, WIRED).is_empty());
        assert_eq!(
            default_findings("pub const DEFAULT_BACKEND: &str = \"obrain\";", WIRED).len(),
            1
        );
        let twice = "#[cfg(not(x))] pub const DEFAULT_BACKEND: &str = \"default\";\n\
                     #[cfg(x)] pub const DEFAULT_BACKEND: &str = \"obrain\";";
        assert_eq!(default_findings(twice, WIRED).len(), 1);
        let hidden = "pub const DEFAULT_BACKEND: &str = \"obrain\";\n\
                      const OLD: &str = r#\"pub const DEFAULT_BACKEND: &str = \"default\";\"#;";
        assert!(!default_findings(hidden, WIRED).is_empty());
        let rewired = "x.unwrap_or_else(|| crate::storage::OBRAIN_BACKEND.to_string())";
        assert_eq!(default_findings(ok, rewired).len(), 1);
        assert!(!default_findings("", WIRED).is_empty());
    }

    #[test]
    fn a_deployment_file_that_selects_obrain_is_reported() {
        assert_eq!(
            deploy_findings("deploy/a.env", "STORAGE_BACKEND=obrain\n").len(),
            1
        );
        assert_eq!(
            deploy_findings("c.yaml", "storage:\n  backend: obrain\n").len(),
            1
        );
        assert!(deploy_findings("c.yaml", "  # backend: obrain\nbackend: default\n").is_empty());
        assert!(is_deploy_file("docker-compose.yml") && is_deploy_file("deploy/prod.env"));
        assert!(!is_deploy_file(".github/workflows/ci.yml") && !is_deploy_file("docs/x.yaml"));
    }

    #[test]
    fn the_obrain_tree_is_scanned_recursively() {
        let dir = std::env::temp_dir().join(format!("migration-gate-{}", std::process::id()));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(
            sub.join("p.rs"),
            "impl ProjectStore for ObrainGraphStore {}",
        )
        .unwrap();
        std::fs::write(
            dir.join("notes.txt"),
            "impl NoteStore for ObrainGraphStore {}",
        )
        .unwrap();
        let mut files = Vec::new();
        rs_files(&dir, &mut files).unwrap();
        let ported: BTreeSet<String> = files
            .iter()
            .flat_map(|p| ported_traits(&std::fs::read_to_string(p).unwrap()))
            .collect();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(ported, set(&["ProjectStore"]));
    }

    #[test]
    fn options_parse_and_an_absent_base_means_no_comparison() {
        let o = parse_args(&["--strict".into(), "--base".into(), "abc".into()]).unwrap();
        assert!(o.strict);
        assert_eq!(o.base.as_deref(), Some("abc"));
        assert!(parse_args(&["--nope".to_string()]).is_err());
        assert!(parse_args(&["--base".to_string()]).is_err());
        assert!(base_is_absent("") && base_is_absent("0000000000"));
        assert!(!base_is_absent("7e1b92dd"));
    }
}
