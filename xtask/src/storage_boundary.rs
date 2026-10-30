//! `cargo xtask storage-boundary` — where do the storage drivers leak out of
//! their adapters? (garde-fou of the storage abstraction)
//!
//! Three pilot drivers — `neo4rs`, `meilisearch_sdk`, `async_nats` — are each
//! owned by one adapter module (`src/neo4j/`, `src/meilisearch/`,
//! `src/events/nats.rs`). Anywhere else, a reference to one of them crosses
//! the storage boundary. This subcommand REPORTS those references; it never
//! rewrites anything.
//!
//! What is scanned:
//! - every `*.rs` file outside the three adapter modules, for a CODE PATH to a
//!   driver (`use …`, a `::` path, a type name) — never a bare substring of a
//!   string literal (regular or raw) or a comment, and never a variable that
//!   merely borrows the driver's name. `src/architecture/*` cites the three
//!   names as data, so it is deliberately never reported;
//! - the `Cargo.toml` of the workspace member crates (`crates/*`), where a
//!   declared dependency on a driver is a leak by declaration. The root
//!   `Cargo.toml` is the main crate — the home of the adapters — so it is not
//!   reported. The `xtask/` crate itself is tooling outside the storage
//!   boundary and is not scanned.
//!
//! Modes: by default it prints the table and the counts and exits 0
//! (informational, « mode avertissement ») ; `--strict` exits 1 while a
//! non-excepted reference remains, or an exception is unused. Exit 2 = could
//! not verify. Exceptions live in `xtask/storage-boundary-exceptions.txt`
//! (one `path:motif  # reason` per line) and can only shrink: an unused
//! exception is reported.

use crate::git::Repo;
use std::path::{Path, PathBuf};

const HELP: &str = "\
usage: cargo xtask storage-boundary [options]

  --repo <path>  backend checkout to scan, default: the one containing the cwd
  --strict       exit 1 while a non-excepted reference remains (or an exception is unused)

exit 0 nothing to report (informational), 1 a reference remains (--strict), 2 could not verify.";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Options {
    pub repo: Option<PathBuf>,
    pub strict: bool,
    pub help: bool,
}

pub fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} attend une valeur"))
        };
        match a.as_str() {
            "--repo" => o.repo = Some(PathBuf::from(value("--repo")?)),
            "--strict" => o.strict = true,
            "-h" | "--help" => o.help = true,
            other => return Err(format!("option inconnue « {other} »")),
        }
    }
    Ok(o)
}

/// One pilot driver crate: its Rust identifier, its Cargo package name, and
/// the adapter module that owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Driver {
    pub rust_name: &'static str,
    pub cargo_name: &'static str,
    pub adapter: &'static str,
}

pub const DRIVERS: &[Driver] = &[
    Driver {
        rust_name: "neo4rs",
        cargo_name: "neo4rs",
        adapter: "src/neo4j/",
    },
    Driver {
        rust_name: "meilisearch_sdk",
        cargo_name: "meilisearch-sdk",
        adapter: "src/meilisearch/",
    },
    Driver {
        rust_name: "async_nats",
        cargo_name: "async-nats",
        adapter: "src/events/nats.rs",
    },
];

/// A driver reference found outside its adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// Path relative to the repo root, `/`-separated.
    pub file: String,
    /// 1-based line number.
    pub line: usize,
    /// Index into [`DRIVERS`].
    pub driver: usize,
    /// The trimmed source line, matched by exception motifs.
    pub text: String,
}

/// One entry of the exceptions file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exception {
    /// File path, or a directory prefix when it ends with `/`.
    pub path: String,
    /// Substring of the cited line; empty = every line of `path`.
    pub motif: String,
    /// The reason (after `#`), mandatory.
    pub reason: String,
    /// 1-based line in the exceptions file.
    pub line: usize,
}

/// Is `path` (relative, `/`-separated) one of the three adapter modules?
pub fn is_adapter_file(path: &str) -> bool {
    path.starts_with("src/neo4j/")
        || path.starts_with("src/meilisearch/")
        || path == "src/events/nats.rs"
}

/// State carried from one line to the next by [`mask_non_code`]: a `/* … */`
/// block comment or a raw string may span lines.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MaskState {
    /// Inside a `/* … */` block comment.
    pub in_block: bool,
    /// Inside a raw string `r"…"` / `r#"…"#` / `br"…"` …; `Some(n)` = expect
    /// exactly `n` `#` right after the closing quote.
    pub raw_hashes: Option<usize>,
}

/// Mask string literals, raw strings, char literals and comments with spaces
/// so that only code remains. `state` carries a block comment or a raw string
/// across lines. A regular string literal is masked within its line (the
/// drivers are never cited in a multi-line regular string here).
pub fn mask_non_code(line: &str, state: &mut MaskState) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if state.in_block {
            if c == '*' && chars.get(i + 1) == Some(&'/') {
                out.push_str("  ");
                i += 2;
                state.in_block = false;
            } else {
                i += 1;
                out.push(' ');
            }
            continue;
        }
        if let Some(n) = state.raw_hashes {
            if c == '"' {
                let mut closes = true;
                for k in 1..=n {
                    if chars.get(i + k) != Some(&'#') {
                        closes = false;
                        break;
                    }
                }
                if closes {
                    out.push(' ');
                    for _ in 0..n {
                        out.push(' ');
                    }
                    i += 1 + n;
                    state.raw_hashes = None;
                } else {
                    out.push(' ');
                    i += 1;
                }
            } else {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        match c {
            '"' => {
                out.push(' ');
                i += 1;
                let mut escaped = false;
                while i < chars.len() {
                    let s = chars[i];
                    i += 1;
                    out.push(' ');
                    if escaped {
                        escaped = false;
                    } else if s == '\\' {
                        escaped = true;
                    } else if s == '"' {
                        break;
                    }
                }
            }
            'r' => {
                // A raw string opener: `r"…"` or `r#…"…"#…` (byte/C raw strings
                // `br`/`cr` share the same `r` prefix, handled identically).
                match chars.get(i + 1) {
                    Some('"') => {
                        out.push_str("  ");
                        i += 2;
                        state.raw_hashes = Some(0);
                    }
                    Some('#') => {
                        let mut n = 0;
                        let mut j = i + 1;
                        while chars.get(j) == Some(&'#') {
                            n += 1;
                            j += 1;
                        }
                        if chars.get(j) == Some(&'"') {
                            out.push(' ');
                            for _ in 0..n {
                                out.push(' ');
                            }
                            out.push(' ');
                            i = j + 1;
                            state.raw_hashes = Some(n);
                        } else {
                            // `r#ident` — a raw identifier, not a string.
                            out.push('r');
                            i += 1;
                        }
                    }
                    _ => {
                        out.push('r');
                        i += 1;
                    }
                }
            }
            '\'' => {
                // Char literal '\x' (4 chars) or 'x' (3 chars); a lifetime
                // 'ident is kept. Distinguishing them matters only so that a
                // '"' char literal does not swallow the rest of the line.
                if chars.get(i + 1) == Some(&'\\') && chars.get(i + 3) == Some(&'\'') {
                    out.push_str("    ");
                    i += 4;
                } else if chars.get(i + 2) == Some(&'\'') {
                    out.push_str("   ");
                    i += 3;
                } else {
                    out.push('\'');
                    i += 1;
                }
            }
            '/' => {
                if chars.get(i + 1) == Some(&'/') {
                    // line comment: the rest of the line is not code
                    out.push(' ');
                    i += 1;
                    while i < chars.len() {
                        out.push(' ');
                        i += 1;
                    }
                } else if chars.get(i + 1) == Some(&'*') {
                    out.push_str("  ");
                    i += 2;
                    state.in_block = true;
                    while i < chars.len() && state.in_block {
                        if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                            out.push_str("  ");
                            i += 2;
                            state.in_block = false;
                        } else {
                            out.push(' ');
                            i += 1;
                        }
                    }
                } else {
                    out.push('/');
                    i += 1;
                }
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// Does the masked code use `name` as a CODE PATH — a crate/module qualifier
/// (`name::…`), a leading absolute path (`::name`), or a bare `use name` —
/// rather than merely as a variable named `name`, a field access `name.field`,
/// or a substring of a longer identifier (`my_name`)?
pub fn uses_driver(code: &str, name: &str) -> bool {
    let mut rest = code;
    while let Some(pos) = rest.find(name) {
        let after = &rest[pos + name.len()..];
        let followed_by_path = after.starts_with("::");
        let preceded_by_path = pos >= 2 && &rest[pos - 2..pos] == "::";
        let preceded_by_use = {
            let before = &rest[..pos];
            let trimmed = before.trim_end();
            if !trimmed.ends_with("use") {
                false
            } else {
                let use_start = trimmed.len() - 3;
                let standalone = use_start == 0 || {
                    let b = trimmed.as_bytes()[use_start - 1];
                    !(b.is_ascii_alphanumeric() || b == b'_')
                };
                // `use` must be separated from `name` by whitespace.
                standalone && before.len() > trimmed.len()
            }
        };
        if followed_by_path || preceded_by_path || preceded_by_use {
            return true;
        }
        rest = &rest[pos + name.len()..];
    }
    false
}

/// The code-path references to a driver in one `.rs` file (whose path is
/// already known to be outside the adapters). One reference per (line, driver).
pub fn rs_references(file: &str, content: &str) -> Vec<Reference> {
    let mut refs = Vec::new();
    let mut state = MaskState::default();
    for (idx, raw) in content.lines().enumerate() {
        let code = mask_non_code(raw, &mut state);
        for (d, driver) in DRIVERS.iter().enumerate() {
            if uses_driver(&code, driver.rust_name) {
                refs.push(Reference {
                    file: file.to_string(),
                    line: idx + 1,
                    driver: d,
                    text: raw.trim().to_string(),
                });
            }
        }
    }
    refs
}

/// The dependency key of a Cargo.toml line (`neo4rs = …` or `"neo4rs" = …`),
/// when the line is a `name = value` pair.
fn toml_dep_key(line: &str) -> Option<String> {
    let (key, _) = line.split_once('=')?;
    let key = key.trim().trim_matches('"');
    (!key.is_empty() && !key.contains(' ')).then(|| key.to_string())
}

/// The driver references of one `Cargo.toml` of a workspace member crate: a
/// dependency whose key is a pilot driver's package name.
pub fn cargo_references(file: &str, content: &str) -> Vec<Reference> {
    let mut refs = Vec::new();
    for (idx, raw) in content.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        let Some(key) = toml_dep_key(line) else {
            continue;
        };
        for (d, driver) in DRIVERS.iter().enumerate() {
            if key == driver.cargo_name {
                refs.push(Reference {
                    file: file.to_string(),
                    line: idx + 1,
                    driver: d,
                    text: raw.trim().to_string(),
                });
            }
        }
    }
    refs
}

/// Parse the exceptions file: one `path:motif  # reason` per line. `path` is a
/// file, or a directory prefix when it ends with `/`; `motif` is a substring
/// of the cited line (empty = every line). Blank lines and `#` comments are
/// skipped. The first `:` separates `path` from `motif`, so a motif may
/// contain `::`.
pub fn parse_exceptions(content: &str) -> Vec<Exception> {
    let mut out = Vec::new();
    for (idx, raw) in content.lines().enumerate() {
        let (spec, reason) = match raw.split_once('#') {
            Some((s, r)) => (s.trim(), r.trim()),
            None => (raw.trim(), ""),
        };
        if spec.is_empty() {
            continue;
        }
        let (path, motif) = match spec.split_once(':') {
            Some((p, m)) => (p.trim(), m.trim()),
            None => (spec, ""),
        };
        out.push(Exception {
            path: path.to_string(),
            motif: motif.to_string(),
            reason: reason.to_string(),
            line: idx + 1,
        });
    }
    out
}

/// Does an exception cover a reference?
fn exception_matches(e: &Exception, r: &Reference) -> bool {
    let path_ok = if e.path.ends_with('/') {
        r.file.starts_with(&e.path)
    } else {
        r.file == e.path
    };
    let motif_ok = e.motif.is_empty() || r.text.contains(&e.motif);
    path_ok && motif_ok
}

/// Split references into excepted / not, and list the exceptions that matched
/// nothing. Returns `(excepted flags parallel to refs, unused exception
/// indices)`.
pub fn apply_exceptions(refs: &[Reference], exceptions: &[Exception]) -> (Vec<bool>, Vec<usize>) {
    let mut excepted = vec![false; refs.len()];
    let mut used = vec![false; exceptions.len()];
    for (ri, r) in refs.iter().enumerate() {
        for (ei, e) in exceptions.iter().enumerate() {
            if exception_matches(e, r) {
                excepted[ri] = true;
                used[ei] = true;
            }
        }
    }
    let unused = used
        .iter()
        .enumerate()
        .filter(|(_, u)| !**u)
        .map(|(i, _)| i)
        .collect();
    (excepted, unused)
}

/// The human-readable report: the `file:line | driver | expected module`
/// table, the count by driver/adapter, the exceptions summary, and the unused
/// exceptions (each of which must be removed, since the list can only shrink).
pub fn render(
    refs: &[Reference],
    excepted: &[bool],
    exceptions: &[Exception],
    unused: &[usize],
) -> String {
    let mut out = String::new();

    if refs.is_empty() {
        out.push_str("aucune référence de pilote hors des adaptateurs\n");
    } else {
        out.push_str(&format!(
            "références de pilote hors des adaptateurs ({}):\n",
            refs.len()
        ));
        for (i, r) in refs.iter().enumerate() {
            let mark = if excepted[i] { "  [exception]" } else { "" };
            out.push_str(&format!(
                "  {}:{} | {} | {}{}\n",
                r.file, r.line, DRIVERS[r.driver].rust_name, DRIVERS[r.driver].adapter, mark
            ));
        }
    }

    let violations = excepted.iter().filter(|e| !**e).count();
    let excepted_count = refs.len() - violations;

    out.push_str("\ndécompte par crate pilote (module attendu) :\n");
    for (d, driver) in DRIVERS.iter().enumerate() {
        let total = refs.iter().filter(|r| r.driver == d).count();
        let non_exc = refs
            .iter()
            .zip(excepted)
            .filter(|(r, e)| r.driver == d && !**e)
            .count();
        out.push_str(&format!(
            "  {:<16} -> {:<20} : {} hors adaptateur, {} non exceptée(s)\n",
            driver.rust_name, driver.adapter, total, non_exc
        ));
    }

    out.push_str(&format!(
        "\n{} référence(s) hors adaptateur : {} non exceptée(s), {} exceptée(s).\n",
        refs.len(),
        violations,
        excepted_count
    ));

    if !unused.is_empty() {
        out.push_str("\nexceptions inutilisées (à retirer du fichier) :\n");
        for &i in unused {
            let e = &exceptions[i];
            out.push_str(&format!(
                "  ligne {} : {}  # {}\n",
                e.line, e.path, e.reason
            ));
        }
    }

    out
}

/// Recursively collect the relative `/`-separated paths of every `*.rs` file
/// under `dir`, skipping directories that are never source (build output,
/// generated code, or the `xtask/` tooling itself).
/// Directories the scan never descends into.
const SKIPPED_DIRS: [&str; 9] = [
    ".git",
    ".rlwt",
    ".wl",
    "_wt",
    "gen",
    "node_modules",
    "target",
    "worktrees",
    "xtask",
];

fn collect_rs_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            // Build output, dependency caches and local working copies of the tree.
            // Counting the copies would multiply every reference by the number of
            // checkouts a developer happens to have; `target-*` covers the per-task
            // build directories this repository uses.
            if SKIPPED_DIRS.contains(&name.as_str()) || name.starts_with("target-") {
                continue;
            }
            collect_rs_files(root, &path, out);
        } else if name.ends_with(".rs") {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// The `Cargo.toml` of every workspace member crate under `crates/` (the root
/// manifest is the main crate — the adapters' home — and is not reported).
fn collect_cargo_files(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let crates_dir = root.join("crates");
    let Ok(entries) = std::fs::read_dir(&crates_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest = path.join("Cargo.toml");
        if manifest.is_file() {
            let rel = manifest.strip_prefix(root).unwrap_or(&manifest);
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    out
}

fn locate_root(o: &Options) -> Result<PathBuf, String> {
    let start = o.repo.clone().unwrap_or_else(|| PathBuf::from("."));
    Repo::discover(&start)
        .map(|r| r.path)
        .ok_or_else(|| format!("{} n'est pas dans un dépôt git", start.display()))
}

/// The only function of this module that touches disk.
pub fn run(args: &[String]) -> i32 {
    let o = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("storage-boundary : {e}\n\n{HELP}");
            return 2;
        }
    };
    if o.help {
        println!("{HELP}");
        return 0;
    }
    let root = match locate_root(&o) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("storage-boundary : non vérifiable — {e}");
            return 2;
        }
    };

    let exceptions_path = root.join("xtask").join("storage-boundary-exceptions.txt");
    let exceptions_content = match std::fs::read_to_string(&exceptions_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "storage-boundary : non vérifiable — {} : {e}",
                exceptions_path.display()
            );
            return 2;
        }
    };
    let exceptions = parse_exceptions(&exceptions_content);

    let mut refs = Vec::new();

    let mut rs_files = Vec::new();
    collect_rs_files(&root, &root, &mut rs_files);
    for file in &rs_files {
        if is_adapter_file(file) {
            continue;
        }
        let path = root.join(file);
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "storage-boundary : non vérifiable — {} : {e}",
                    path.display()
                );
                return 2;
            }
        };
        refs.extend(rs_references(file, &content));
    }

    for file in collect_cargo_files(&root) {
        let path = root.join(&file);
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "storage-boundary : non vérifiable — {} : {e}",
                    path.display()
                );
                return 2;
            }
        };
        refs.extend(cargo_references(&file, &content));
    }

    refs.sort_by(|a, b| (a.file.as_str(), a.line).cmp(&(b.file.as_str(), b.line)));

    let (excepted, unused) = apply_exceptions(&refs, &exceptions);
    print!("{}", render(&refs, &excepted, &exceptions, &unused));

    let violations = excepted.iter().filter(|e| !**e).count();
    if o.strict && (violations > 0 || !unused.is_empty()) {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_adapter_file_recognizes_only_the_three_modules() {
        assert!(is_adapter_file("src/neo4j/client.rs"));
        assert!(is_adapter_file("src/meilisearch/client.rs"));
        assert!(is_adapter_file("src/events/nats.rs"));
        assert!(!is_adapter_file("src/events/nats_broker_test.rs"));
        assert!(!is_adapter_file("src/events/bus.rs"));
        assert!(!is_adapter_file("src/chat/manager.rs"));
        assert!(!is_adapter_file("src/neo4js/util.rs"));
    }

    #[test]
    fn mask_non_code_strips_strings_line_and_block_comments() {
        let mut state = MaskState::default();
        let line = r#"use neo4rs::Graph; // "meilisearch_sdk""#;
        let masked = mask_non_code(line, &mut state);
        assert_eq!(masked.len(), line.len(), "le masquage conserve la longueur");
        assert!(masked.contains("use neo4rs::Graph;"), "{masked}");
        assert!(!masked.contains("meilisearch_sdk"), "{masked}");

        let line = r#"let s = "async_nats"; /* meilisearch_sdk */"#;
        let masked = mask_non_code(line, &mut state);
        assert_eq!(masked.len(), line.len());
        assert!(masked.contains("let s = "), "{masked}");
        assert!(!masked.contains("async_nats"), "{masked}");
        assert!(!masked.contains("meilisearch_sdk"), "{masked}");
    }

    #[test]
    fn mask_non_code_masks_raw_strings_across_lines() {
        let lines = [
            r#"const CARGO: &str = r#""#,
            "[package]",
            r#"neo4rs = "0.8""#,
            "\"#;",
        ];
        let mut state = MaskState::default();
        let mut masked = String::new();
        for l in lines {
            masked.push_str(&mask_non_code(l, &mut state));
            masked.push('\n');
        }
        assert!(!uses_driver(&masked, "neo4rs"), "{masked}");
        assert_eq!(state.raw_hashes, None);
    }

    #[test]
    fn mask_non_code_handles_char_literals_without_swallowing_the_line() {
        let mut state = MaskState::default();
        // A '"' char literal must not open a string that eats the rest.
        let masked = mask_non_code(r#"let q = '"'; use neo4rs::Graph;"#, &mut state);
        assert!(uses_driver(&masked, "neo4rs"), "{masked:?}");
        assert!(!masked.contains('"'));
    }

    #[test]
    fn mask_non_code_tracks_block_comments_across_lines() {
        let mut state = MaskState::default();
        let l1 = "let a = 1; /*";
        let m1 = mask_non_code(l1, &mut state);
        assert_eq!(m1.len(), l1.len());
        assert!(m1.contains("let a = 1;"), "{m1}");
        assert!(state.in_block);

        let l2 = "still a comment */ let b = 2;";
        let m2 = mask_non_code(l2, &mut state);
        assert_eq!(m2.len(), l2.len());
        assert!(m2.contains("let b = 2;"), "{m2}");
        assert!(!state.in_block);
    }

    #[test]
    fn uses_driver_requires_a_code_path_not_a_bare_identifier() {
        // A code path: `use …`, `::`, type names.
        assert!(uses_driver("use neo4rs::{query, Graph};", "neo4rs"));
        assert!(uses_driver("use neo4rs;", "neo4rs"));
        assert!(uses_driver("let g = neo4rs::Graph::new();", "neo4rs"));
        assert!(uses_driver("::neo4rs::Graph", "neo4rs"));
        // A variable, a field access, a longer identifier: not a code path.
        assert!(!uses_driver("let neo4rs = deps.find();", "neo4rs"));
        assert!(!uses_driver("neo4rs.line", "neo4rs"));
        assert!(!uses_driver("my_neo4rs_helper", "neo4rs"));
        assert!(!uses_driver("useneo4rs", "neo4rs"));
        assert!(!uses_driver("abuse neo4rs", "neo4rs"));
    }

    #[test]
    fn rs_references_finds_code_paths_and_skips_data() {
        let src = r#"// a comment citing neo4rs
use neo4rs::{query, Graph};
let name = "neo4rs";
let g = neo4rs::Graph::new();
"#;
        let refs = rs_references("src/x.rs", src);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].line, 2);
        assert_eq!(refs[1].line, 4);
        assert!(refs.iter().all(|r| r.driver == 0));
    }

    #[test]
    fn architecture_data_is_never_reported() {
        // `src/architecture/*` cites the three names as data: a doc comment, a
        // line comment, a local variable named `neo4rs`, a field access on it,
        // and plain string literals. None is a code path, so none is reported.
        let src = r#"/// The package that implied this, e.g. "neo4rs".
fn demo(deps: &[Dependency]) -> Option<Dependency> {
    // recognising `neo4rs` as a driver and `async_nats` as a bus
    let neo4rs = deps.iter().find(|d| d.name == "neo4rs").unwrap();
    assert_eq!(neo4rs.line, 7);
    assert_eq!(neo4rs.manifest_path, "Cargo.toml");
    if name == "meilisearch_sdk" { return None; }
    None
}
"#;
        assert!(rs_references("src/architecture/catalogue.rs", src).is_empty());
    }

    #[test]
    fn cargo_references_reads_a_driver_dependency() {
        let src = "[dependencies]\nserde = { workspace = true }\nneo4rs = { workspace = true }\n# meilisearch-sdk = \"0.33\"\n\"async-nats\" = \"0.49\"\n";
        let refs = cargo_references("crates/x/Cargo.toml", src);
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].line, 3);
        assert_eq!(refs[0].driver, 0); // neo4rs
        assert_eq!(refs[1].line, 5);
        assert_eq!(refs[1].driver, 2); // async-nats
    }

    #[test]
    fn parse_exceptions_reads_path_motif_and_reason() {
        let src = "# header\ntests/:  # integration tests\nsrc/chat/manager.rs:meilisearch_sdk::client  # direct client\n";
        let excs = parse_exceptions(src);
        assert_eq!(excs.len(), 2);
        assert_eq!(excs[0].path, "tests/");
        assert_eq!(excs[0].motif, "");
        assert_eq!(excs[0].reason, "integration tests");
        assert_eq!(excs[0].line, 2);
        assert_eq!(excs[1].path, "src/chat/manager.rs");
        assert_eq!(excs[1].motif, "meilisearch_sdk::client");
        assert_eq!(excs[1].reason, "direct client");
    }

    #[test]
    fn apply_exceptions_reports_unused_and_flags_matches() {
        let refs = vec![
            Reference {
                file: "tests/a.rs".into(),
                line: 1,
                driver: 0,
                text: "neo4rs::Graph::new()".into(),
            },
            Reference {
                file: "src/chat/manager.rs".into(),
                line: 9,
                driver: 1,
                text: "meilisearch_sdk::client::Client".into(),
            },
        ];
        let excs = vec![
            Exception {
                path: "tests/".into(),
                motif: "".into(),
                reason: "tests".into(),
                line: 2,
            },
            Exception {
                path: "src/chat/manager.rs".into(),
                motif: "meilisearch_sdk::client".into(),
                reason: "direct client".into(),
                line: 3,
            },
            Exception {
                path: "src/api/handlers.rs".into(),
                motif: "async_nats".into(),
                reason: "unused".into(),
                line: 4,
            },
        ];
        let (excepted, unused) = apply_exceptions(&refs, &excs);
        assert_eq!(excepted, vec![true, true]);
        assert_eq!(unused, vec![2]);
    }

    #[test]
    fn parse_args_reads_strict_and_repo() {
        let args: Vec<String> = ["--strict", "--repo", "/x"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let o = parse_args(&args).unwrap();
        assert!(o.strict);
        assert_eq!(o.repo, Some(PathBuf::from("/x")));
        assert!(parse_args(&["--bogus".to_string()]).is_err());
        assert!(parse_args(&["--repo".to_string()]).is_err());
    }

    #[test]
    fn render_shows_table_counts_and_unused_exceptions() {
        let refs = vec![Reference {
            file: "src/chat/manager.rs".into(),
            line: 42,
            driver: 1,
            text: "meilisearch_sdk::client::Client".into(),
        }];
        let excepted = vec![true];
        let excs = vec![Exception {
            path: "tests/".into(),
            motif: "".into(),
            reason: "tests".into(),
            line: 2,
        }];
        let text = render(&refs, &excepted, &excs, &[0]);
        assert!(
            text.contains("src/chat/manager.rs:42 | meilisearch_sdk | src/meilisearch/"),
            "{text}"
        );
        assert!(text.contains("[exception]"), "{text}");
        assert!(text.contains("0 non exceptée(s)"), "{text}");
        assert!(text.contains("exceptions inutilisées"), "{text}");
    }
}
