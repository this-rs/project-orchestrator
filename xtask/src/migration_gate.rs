//! `cargo xtask migration-gate` — the part of the Neo4j → obrain migration gate
//! that needs `git`, not the compiler. See `docs/migration/GATES.md`.
//!
//! Which store traits are "ported" is decided by the build generator
//! (`build/refus_obrain.rs`, parsed with `syn`), and the test
//! `the_migration_ratchet_is_exactly_the_set_of_ported_traits`
//! (`tests/obrain_backend.rs`, job « Obrain backend ») checks that the ratchet
//! `xtask/migration-portees.txt` is exactly that set. A text search here would
//! disagree with the compiler, so this subcommand does not try to.
//!
//! What it checks:
//!
//! 1. **The ratchet never shrinks (G2).** With `--base <rev>`, a trait listed in
//!    the ratchet at `<rev>` and missing now is refused (`CLIQUET RÉDUIT`).
//!    Removing a port AND its ratchet line in the same change passes the test
//!    above (both sides shrink together) and is caught here.
//!
//! 2. **The default backend stays Neo4j (G3)**, a tripwire on the source, not a
//!    behavioural proof. Directive of 20/11/2026: « on ne branche pas en prod
//!    tant qu'on n'a pas migré vers obrain et this-rs ». Refused when:
//!    - `DEFAULT_BACKEND` is not defined exactly once, with the value `"default"`;
//!    - the storage factory no longer builds the Neo4j backend for
//!      `BackendKind::Default`;
//!    - the configuration (`src/lib.rs`) no longer falls back to
//!      `DEFAULT_BACKEND` in code (a copy in a string does not count);
//!    - a tracked deployment file sets `STORAGE_BACKEND` (or a YAML `backend:`)
//!      to obrain.
//!
//!    Known blind spots, listed in GATES.md: a fallback to obrain placed before
//!    the default in `src/lib.rs`, a copy of the wiring in dead code, and a
//!    deployment value reached through an indirection (`ARG` then `ENV $VAR`).
//!
//! Exit codes: 0 nothing to report, or informational mode; 1 a finding
//! (`--strict`); 2 could not verify (not a repository, an unreadable file, an
//! unknown `--base`, the ratchet missing at a base that has it).

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::git::Repo;

const HELP: &str = "\
usage: cargo xtask migration-gate [--strict] [--base <rev>] [--repo <chemin>]

  --strict   exit 1 on any finding: the ratchet shrank since --base, or the
             default backend is no longer Neo4j
  --base     compare the ratchet with the one at this revision (the target
             branch of a pull request). Empty or all-zero = no comparison.
  --repo     start the repository discovery from this path (default: .)

Which traits are ported is checked by the test
the_migration_ratchet_is_exactly_the_set_of_ported_traits (tests/obrain_backend.rs).

exit 0 nothing to report (or informational mode), 1 a finding (--strict),
2 could not verify.";

const RATCHET_FILE: &str = "xtask/migration-portees.txt";
/// The storage module (a file before the storage split, a directory after).
const STORAGE_FILES: &[&str] = &["src/storage/mod.rs", "src/storage.rs"];
const CONFIG_FILE: &str = "src/lib.rs";
/// How the configuration falls back to the default backend.
const DEFAULT_WIRING: &str = "unwrap_or_else(|| crate::storage::DEFAULT_BACKEND.to_string())";
/// How the storage factory builds the backend for the default kind.
const DEFAULT_ARM: &str = "BackendKind::Default => Self::default_backend(";

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Remove comments (line, nested block) and, when `strings` is false, the
/// contents of string literals (regular, byte, raw, C). Removed text becomes
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
        if ch == '/' && next == Some('/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
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
        // Raw string r"…", r#"…"#, and its byte / C forms br"…", cr"…": the `r`
        // must not be inside an identifier, except a single `b` or `c` prefix.
        let prefix_ok = |k: usize| k < 2 || !is_ident(c[k - 2]);
        let prev_ident =
            i > 0 && is_ident(c[i - 1]) && !((c[i - 1] == 'b' || c[i - 1] == 'c') && prefix_ok(i));
        if ch == 'r' && !prev_ident {
            let mut j = i + 1;
            while c.get(j) == Some(&'#') {
                j += 1;
            }
            if c.get(j) == Some(&'"') {
                let hashes = j - i - 1;
                out.extend(&c[i..=j]);
                i = j + 1;
                while i < c.len() {
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

/// The ratchet file: one trait per line, `#` starts a comment.
pub fn ratchet(content: &str) -> BTreeSet<String> {
    content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Traits listed in the base ratchet and missing from the current one.
pub fn shrunk(base: &BTreeSet<String>, current: &BTreeSet<String>) -> Vec<String> {
    base.difference(current).cloned().collect()
}

/// Findings on the default backend; empty = conforme.
pub fn default_findings(storage: &str, config: &str) -> Vec<String> {
    let mut f = Vec::new();
    // The definition: comments removed, strings KEPT, so a definition hidden
    // in a string counts as a second one and is refused (fail-closed).
    let with_strings = flatten(&strip_code(storage, true));
    let defs: Vec<&str> = with_strings
        .match_indices("const DEFAULT_BACKEND:")
        .map(|(i, _)| &with_strings[i..])
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
    // The factory arm and the configuration wiring: strings REMOVED, so a copy
    // kept in a string does not stand for the real code.
    let code = flatten(&strip_code(storage, false));
    if !code.contains(&flatten(DEFAULT_ARM)) {
        f.push(
            "la fabrique de stockage ne construit plus le backend Neo4j pour BackendKind::Default"
                .to_string(),
        );
    }
    let cfg = flatten(&strip_code(config, false));
    if !cfg.contains(&flatten(DEFAULT_WIRING)) {
        f.push(format!(
            "{CONFIG_FILE} ne retombe plus sur DEFAULT_BACKEND quand rien n'est configuré"
        ));
    }
    f
}

/// Does this lowercase line set a YAML `backend:` key? The key must start the
/// line or follow a space, `{` or `,` — so `obrain_backend:` and a CI job named
/// `obrain-backend:` are not a backend setting.
fn sets_backend_key(l: &str) -> bool {
    l.match_indices("backend:").any(|(i, _)| {
        l[..i]
            .chars()
            .last()
            .is_none_or(|c| c.is_whitespace() || c == '{' || c == ',')
    })
}

/// Lines of a deployment file that select obrain as the storage backend,
/// including a Kubernetes-style `- name: STORAGE_BACKEND` / `value: obrain` pair.
pub fn deploy_findings(path: &str, content: &str) -> Vec<String> {
    let lines: Vec<String> = content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim().to_lowercase())
        .collect();
    let mut out = Vec::new();
    for (n, l) in lines.iter().enumerate() {
        let env = l.contains("storage_backend") && l.contains("obrain");
        let yaml = sets_backend_key(l) && l.contains("obrain");
        let pair = l.contains("storage_backend")
            && !l.contains("obrain")
            && lines[n + 1..]
                .iter()
                .find(|x| !x.is_empty())
                .is_some_and(|x| x.starts_with("value:") && x.contains("obrain"));
        if env || yaml || pair {
            out.push(format!("{path}:{} sélectionne obrain", n + 1));
        }
    }
    out
}

/// Is this tracked file a deployment configuration to scan?
fn is_deploy_file(path: &str) -> bool {
    // Rust sources, tests, docs, benches and the tooling are not deployment.
    let skip = [
        "docs/",
        "tests/",
        "examples/",
        "xtask/",
        "src/",
        "build/",
        "tools/",
        "benches/",
    ];
    if skip.iter().any(|p| path.starts_with(p)) || path.ends_with(".md") || path.ends_with(".rs") {
        return false;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    let name = name.strip_suffix(".example").unwrap_or(name);
    name.ends_with(".yml")
        || name.ends_with(".yaml")
        || name.ends_with(".env")
        || name.starts_with(".env")
        || name.ends_with(".service")
        || name.ends_with(".sh")
        || name.ends_with(".toml")
        || name.starts_with("Dockerfile")
        || name.starts_with("docker-compose")
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

    let current = match read(RATCHET_FILE) {
        Ok(t) => ratchet(&t),
        Err(e) => return unverifiable(e),
    };

    let base = match o.base.as_deref() {
        None => None,
        Some(rev) if base_is_absent(rev) => None,
        Some(rev) => {
            if repo.resolve(rev).is_none() {
                return unverifiable(format!("révision de base « {rev} » inconnue"));
            }
            // Is the ratchet tracked at the base? Absent there (the gate is new)
            // means nothing to compare; present but unreadable is not verifiable.
            let tracked = repo
                .git(&["ls-tree", "--name-only", rev, "--", RATCHET_FILE])
                .map(|out| out.lines().any(|l| l.trim() == RATCHET_FILE));
            match tracked {
                Err(e) => return unverifiable(format!("git ls-tree {rev} : {e}")),
                Ok(false) => Some(BTreeSet::new()),
                Ok(true) => {
                    let spec = format!("{rev}:{RATCHET_FILE}");
                    match repo.git(&["show", &spec]) {
                        Ok(t) => Some(ratchet(&t)),
                        Err(e) => return unverifiable(format!("git show {spec} : {e}")),
                    }
                }
            }
        }
    };
    let removed = base
        .as_ref()
        .map(|b| shrunk(b, &current))
        .unwrap_or_default();

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

    println!(
        "migration-gate : {} trait(s) dans le cliquet{}",
        current.len(),
        match &base {
            Some(b) => format!(", {} dans le cliquet de base", b.len()),
            None => " (pas de base : pas de comparaison)".to_string(),
        }
    );
    for t in &removed {
        println!("  CLIQUET RÉDUIT   {t} : retiré de {RATCHET_FILE} depuis la base");
    }
    if defaults.is_empty() {
        println!("  backend par défaut : neo4j (« default »), câblé — conforme");
    } else {
        for d in &defaults {
            println!("  BACKEND PAR DÉFAUT : {d} — refusé");
        }
    }

    let failed = !removed.is_empty() || !defaults.is_empty();
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
    const ARM: &str = "match k { BackendKind::Default => Self::default_backend(config).await, }";

    fn storage(def: &str) -> String {
        format!("{def}\n{ARM}")
    }

    #[test]
    fn comments_and_strings_are_removed_but_lines_kept() {
        let src = "a // impl X for ObrainA {\nb /* impl Y /* nested */ for ObrainB { */ c\n\
                   let s = \"impl Z for ObrainC {\"; let r = r#\"impl W for ObrainD {\"#;\n\
                   let k = cr#\"impl V for ObrainE {\"#;\n\
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
    fn the_ratchet_ignores_comments_and_blank_lines() {
        assert_eq!(
            ratchet("# en-tête\n\nNoteStore  # porté\nPlanStore\n"),
            set(&["NoteStore", "PlanStore"])
        );
    }

    #[test]
    fn a_line_removed_since_the_base_is_reported() {
        let base = set(&["ProjectStore", "NoteStore"]);
        assert_eq!(
            shrunk(&base, &set(&["NoteStore"])),
            vec!["ProjectStore".to_string()]
        );
        assert!(shrunk(&base, &set(&["NoteStore", "ProjectStore", "PlanStore"])).is_empty());
    }

    #[test]
    fn the_default_backend_is_checked_by_definition_factory_and_wiring() {
        let ok = storage("pub const DEFAULT_BACKEND: &str = \"default\"; // neo4j");
        assert!(
            default_findings(&ok, WIRED).is_empty(),
            "{:?}",
            default_findings(&ok, WIRED)
        );
        let wrapped = storage("pub const DEFAULT_BACKEND: &str =\n    \"default\";");
        assert!(default_findings(&wrapped, WIRED).is_empty());
        let obrain = storage("pub const DEFAULT_BACKEND: &str = \"obrain\";");
        assert_eq!(default_findings(&obrain, WIRED).len(), 1);
        let twice = storage(
            "#[cfg(not(x))] pub const DEFAULT_BACKEND: &str = \"default\";\n\
             #[cfg(x)] pub const DEFAULT_BACKEND: &str = \"obrain\";",
        );
        assert_eq!(default_findings(&twice, WIRED).len(), 1);
        let hidden = storage(
            "pub const DEFAULT_BACKEND: &str = \"obrain\";\n\
             const OLD: &str = r#\"pub const DEFAULT_BACKEND: &str = \"default\";\"#;",
        );
        assert!(!default_findings(&hidden, WIRED).is_empty());
        // The factory arm rewired to obrain.
        let rearmed = "pub const DEFAULT_BACKEND: &str = \"default\";\n\
                       BackendKind::Default => Self::obrain_backend(config).await,";
        assert_eq!(default_findings(rearmed, WIRED).len(), 1);
        // The wiring replaced, with a copy left in a string.
        let rewired = "x.unwrap_or_else(|| crate::storage::OBRAIN_BACKEND.to_string()); \
                       let _s = \"unwrap_or_else(|| crate::storage::DEFAULT_BACKEND.to_string())\";";
        assert_eq!(default_findings(&ok, rewired).len(), 1);
        // A label that merely starts with the name is not a second definition.
        let label = storage(
            "pub const DEFAULT_BACKEND: &str = \"default\";\n\
             pub const DEFAULT_BACKEND_LABEL: &str = \"Neo4j\";",
        );
        assert!(default_findings(&label, WIRED).is_empty());
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
        assert_eq!(
            deploy_findings("c.yaml", "storage: { backend: obrain }\n").len(),
            1
        );
        assert_eq!(
            deploy_findings(
                "k.yaml",
                "env:\n  - name: STORAGE_BACKEND\n    value: obrain\n"
            )
            .len(),
            1
        );
        assert_eq!(
            deploy_findings("x.service", "Environment=STORAGE_BACKEND=obrain\n").len(),
            1
        );
        assert!(deploy_findings("c.yaml", "  # backend: obrain\nbackend: default\n").is_empty());
        assert!(deploy_findings("c.yaml", "obrain_backend: yes\n").is_empty());
        assert!(deploy_findings(
            "ci.yml",
            "jobs:\n  obrain-backend:\n    name: Obrain backend\n"
        )
        .is_empty());
        assert!(is_deploy_file("docker-compose.yml") && is_deploy_file("deploy/prod.env"));
        assert!(is_deploy_file("config.yaml.example") && is_deploy_file("install.sh"));
        assert!(is_deploy_file(
            "packaging/debian/project-orchestrator.service"
        ));
        assert!(is_deploy_file(".github/workflows/release.yml"));
        assert!(!is_deploy_file("docs/x.yaml") && !is_deploy_file("tools/bench/bench.env"));
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
