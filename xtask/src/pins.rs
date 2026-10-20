//! `cargo xtask pins` — did a merge in one repo reach the other? (AM-2)
//!
//! For every dependency Cargo.toml pins by git `rev`, says whether that sha
//! is the head of the repo's main, a sha of main N commits back (and which
//! merges it lacks), or a sha main does not contain at all (an intermediate
//! or branch sha). Repos it cannot verify are NAMED, never skipped.
//!
//! Works from local clones so it replays offline: `--clone <repo>=<path>`,
//! else the sibling directory `../<repo>` of the backend checkout (the
//! layout of this machine: backend/ and nexus/ side by side). `git fetch` of
//! the clone is on by default and optional (`--no-fetch`).

use crate::git::{remote_of, short, Commit, Repo};
use std::path::{Path, PathBuf};

const HELP: &str = "\
usage: cargo xtask pins [options]

  --manifest <file>      Cargo.toml to read, default: the one at the repo root
  --repo <path>          backend checkout, default: the one containing the cwd
  --clone <name>=<path>  local clone of a pinned repo (repeatable); default ../<name>
  --ref <ref>            what « main » means in each clone, default origin/main
  --no-fetch             do not `git fetch` the clones first (offline replay)

exit 0 every pin is main's head, 1 a pin is behind or off main, 2 a repo could not be verified.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub manifest: Option<PathBuf>,
    pub repo: Option<PathBuf>,
    pub clones: Vec<(String, PathBuf)>,
    pub reference: String,
    pub fetch: bool,
    pub help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            manifest: None,
            repo: None,
            clones: Vec::new(),
            reference: "origin/main".to_string(),
            fetch: true,
            help: false,
        }
    }
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
            "--manifest" => o.manifest = Some(PathBuf::from(value("--manifest")?)),
            "--repo" => o.repo = Some(PathBuf::from(value("--repo")?)),
            "--clone" => {
                let v = value("--clone")?;
                let (name, path) = v
                    .split_once('=')
                    .ok_or_else(|| format!("--clone attend <nom>=<chemin>, reçu « {v} »"))?;
                o.clones.push((name.to_string(), PathBuf::from(path)));
            }
            "--ref" => o.reference = value("--ref")?,
            "--no-fetch" => o.fetch = false,
            "-h" | "--help" => o.help = true,
            other => return Err(format!("option inconnue « {other} »")),
        }
    }
    Ok(o)
}

/// One `git = "<url>", rev = "<sha>"` dependency of a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pin {
    pub dep: String,
    pub url: String,
    pub rev: String,
    pub line: usize,
}

/// Value of `key = "..."` inside one TOML line (inline table or plain
/// key/value); `None` when the key is absent.
fn toml_str(line: &str, key: &str) -> Option<String> {
    let mut rest = line;
    while let Some(pos) = rest.find(key) {
        // `rev` must start a key: not the tail of `brev` or `my_rev`.
        let before_ok = pos == 0 || {
            let prev = rest.as_bytes()[pos - 1];
            !(prev.is_ascii_alphanumeric() || prev == b'-' || prev == b'_')
        };
        let after = &rest[pos + key.len()..];
        let after_trim = after.trim_start();
        if before_ok {
            if let Some(v) = after_trim.strip_prefix('=') {
                let v = v.trim_start();
                if let Some(s) = v.strip_prefix('"') {
                    let end = s.find('"')?;
                    return Some(s[..end].to_string());
                }
                return None;
            }
        }
        rest = after;
    }
    None
}

/// Dependency name of a line (`nexus-claude = { ... }` or `"nexus-claude" = ...`).
fn dep_name(line: &str) -> Option<String> {
    let (key, _) = line.split_once('=')?;
    let key = key.trim().trim_matches('"');
    (!key.is_empty() && !key.contains(' ')).then(|| key.to_string())
}

/// Every git+rev pin of a manifest, in file order. Reads inline tables
/// (`name = { git = "...", rev = "..." }`) and `[*dependencies.name]` tables
/// whose `git`/`rev` keys are on their own lines; `[patch]` sections too,
/// since a patch pins just the same.
pub fn parse_pins(manifest: &str) -> Vec<Pin> {
    let mut pins = Vec::new();
    // Current `[*dependencies.<name>]` table: (name, line, git, rev).
    let mut table: Option<(String, usize, Option<String>, Option<String>)> = None;
    let flush = |t: &mut Option<(String, usize, Option<String>, Option<String>)>,
                 pins: &mut Vec<Pin>| {
        if let Some((dep, line, Some(url), Some(rev))) = t.take() {
            pins.push(Pin {
                dep,
                url,
                rev,
                line,
            });
        }
    };
    for (i, raw) in manifest.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            flush(&mut table, &mut pins);
            let header = line.trim_matches(|c| c == '[' || c == ']');
            table = header
                .rsplit_once('.')
                .filter(|(section, _)| {
                    section.ends_with("dependencies") || section.starts_with("patch")
                })
                .map(|(_, name)| (name.trim_matches('"').to_string(), i + 1, None, None));
            continue;
        }
        if let (Some(url), Some(rev)) = (toml_str(line, "git"), toml_str(line, "rev")) {
            if let Some(dep) = dep_name(line) {
                pins.push(Pin {
                    dep,
                    url,
                    rev,
                    line: i + 1,
                });
            }
            continue;
        }
        if let Some(t) = table.as_mut() {
            if let Some(key) = dep_name(line) {
                match key.as_str() {
                    "git" => t.2 = toml_str(line, "git"),
                    "rev" => t.3 = toml_str(line, "rev"),
                    _ => {}
                }
            }
        }
    }
    flush(&mut table, &mut pins);
    pins
}

/// `https://github.com/this-rs/nexus.git` → `nexus`.
pub fn repo_name(url: &str) -> String {
    url.trim_end_matches('/')
        .rsplit(['/', ':'])
        .next()
        .unwrap_or(url)
        .trim_end_matches(".git")
        .to_string()
}

/// The pins of one repository, grouped by distinct rev (file order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoPins {
    pub name: String,
    pub url: String,
    pub revs: Vec<RevUse>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevUse {
    pub rev: String,
    pub deps: Vec<String>,
}

pub fn group_by_repo(pins: &[Pin]) -> Vec<RepoPins> {
    let mut repos: Vec<RepoPins> = Vec::new();
    for p in pins {
        let name = repo_name(&p.url);
        let repo = match repos.iter_mut().find(|r| r.name == name) {
            Some(r) => r,
            None => {
                repos.push(RepoPins {
                    name,
                    url: p.url.clone(),
                    revs: Vec::new(),
                });
                repos.last_mut().expect("just pushed")
            }
        };
        match repo.revs.iter_mut().find(|r| r.rev == p.rev) {
            Some(r) => {
                if !r.deps.contains(&p.dep) {
                    r.deps.push(p.dep.clone());
                }
            }
            None => repo.revs.push(RevUse {
                rev: p.rev.clone(),
                deps: vec![p.dep.clone()],
            }),
        }
    }
    repos
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinVerdict {
    Unverifiable(String),
    /// The pin is main's head.
    UpToDate {
        main: String,
    },
    /// The pin is on main, these merges (newest first) came after it.
    Behind {
        main: String,
        merges: Vec<Commit>,
    },
    /// The pin is not on main: an intermediate or branch sha.
    OffMain {
        main: String,
        /// Merges of main the pin does not contain.
        merges: Vec<Commit>,
        /// Commits the pin has that main does not.
        commits_off: usize,
    },
}

impl PinVerdict {
    pub fn exit_code(&self) -> i32 {
        match self {
            PinVerdict::UpToDate { .. } => 0,
            PinVerdict::Behind { .. } | PinVerdict::OffMain { .. } => 1,
            PinVerdict::Unverifiable(_) => 2,
        }
    }
}

/// Verdict of one pinned rev against `main_ref` in the repo's clone (git reads only).
pub fn check_rev(clone: &Repo, main_ref: &str, rev: &str) -> PinVerdict {
    let Some(main) = clone.resolve(main_ref) else {
        return PinVerdict::Unverifiable(format!(
            "« {main_ref} » n'est pas un commit du clone {} (git fetch ?)",
            clone.path.display()
        ));
    };
    let Some(pinned) = clone.resolve(rev) else {
        return PinVerdict::Unverifiable(format!(
            "sha {} inconnu du clone {} (git fetch ?)",
            short(rev),
            clone.path.display()
        ));
    };
    let merges = match clone.merges_between(&pinned, &main) {
        Ok(m) => m,
        Err(e) => return PinVerdict::Unverifiable(e),
    };
    if clone.is_ancestor(&pinned, &main) {
        if merges.is_empty() && pinned == main {
            PinVerdict::UpToDate { main }
        } else {
            PinVerdict::Behind { main, merges }
        }
    } else {
        match clone.count_between(&main, &pinned) {
            Ok(commits_off) => PinVerdict::OffMain {
                main,
                merges,
                commits_off,
            },
            Err(e) => PinVerdict::Unverifiable(e),
        }
    }
}

/// French agreement: 0 and 1 take the singular (« 0 fusion », « 1 commit »).
fn plural(n: usize, one: &str, many: &str) -> String {
    if n <= 1 {
        format!("{n} {one}")
    } else {
        format!("{n} {many}")
    }
}

/// « <repo> main ↔ épingle : ... », one block per distinct rev.
pub fn render_repo(repo: &RepoPins, verdicts: &[PinVerdict]) -> String {
    let mut out = String::new();
    if repo.revs.len() > 1 {
        out.push_str(&format!(
            "{} : {} épingles différentes dans Cargo.toml ({})\n",
            repo.name,
            repo.revs.len(),
            repo.revs
                .iter()
                .map(|r| format!("{} = {}", r.deps.join(", "), short(&r.rev)))
                .collect::<Vec<_>>()
                .join(" ; ")
        ));
    }
    for (use_, verdict) in repo.revs.iter().zip(verdicts) {
        let head = format!(
            "{} main ↔ épingle {} ({})",
            repo.name,
            short(&use_.rev),
            use_.deps.join(", ")
        );
        match verdict {
            PinVerdict::Unverifiable(why) => {
                out.push_str(&format!("{head} : non vérifiable — {why}\n"));
            }
            PinVerdict::UpToDate { main } => {
                out.push_str(&format!("{head} : à jour (main = {})\n", short(main)));
            }
            PinVerdict::Behind { main, merges } => {
                out.push_str(&format!(
                    "{head} : {} de retard sur main {} (sha de main)\n",
                    plural(merges.len(), "commit", "commits"),
                    short(main)
                ));
                for m in merges {
                    out.push_str(&format!("  - {} {}\n", short(&m.sha), m.subject));
                }
            }
            PinVerdict::OffMain {
                main,
                merges,
                commits_off,
            } => {
                out.push_str(&format!(
                    "{head} : sha intermédiaire, hors de main {} ({} que main n'a pas ; main a {} que l'épingle n'a pas)\n",
                    short(main),
                    plural(*commits_off, "commit", "commits"),
                    plural(merges.len(), "commit", "commits"),
                ));
                for m in merges {
                    out.push_str(&format!("  - {} {}\n", short(&m.sha), m.subject));
                }
            }
        }
    }
    out
}

/// Where the clone of `name` is: an explicit `--clone`, else `<root>/../<name>`.
pub fn clone_path(name: &str, clones: &[(String, PathBuf)], root: &Path) -> PathBuf {
    clones
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, p)| p.clone())
        .unwrap_or_else(|| root.join("..").join(name))
}

fn locate_repo(o: &Options) -> Result<Repo, String> {
    match &o.repo {
        Some(p) => {
            let r = Repo::at(p);
            r.is_repo()
                .then_some(r)
                .ok_or_else(|| format!("{} n'est pas un dépôt git", p.display()))
        }
        None => Repo::discover(Path::new(".")).ok_or_else(|| {
            "le répertoire courant n'est pas dans un dépôt git (--repo <chemin>)".to_string()
        }),
    }
}

/// The only function of this module that touches disk and network.
pub fn run(args: &[String]) -> i32 {
    let o = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("pins : {e}\n\n{HELP}");
            return 2;
        }
    };
    if o.help {
        println!("{HELP}");
        return 0;
    }
    let root = match locate_repo(&o) {
        Ok(r) => r.path,
        Err(e) => {
            eprintln!("pins : non vérifiable — {e}");
            return 2;
        }
    };
    let manifest_path = o
        .manifest
        .clone()
        .unwrap_or_else(|| root.join("Cargo.toml"));
    let manifest = match std::fs::read_to_string(&manifest_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("pins : non vérifiable — {} : {e}", manifest_path.display());
            return 2;
        }
    };
    let repos = group_by_repo(&parse_pins(&manifest));
    if repos.is_empty() {
        println!(
            "aucune dépendance épinglée par git rev dans {}",
            manifest_path.display()
        );
        return 0;
    }
    let mut code = 0;
    for repo in &repos {
        let path = clone_path(&repo.name, &o.clones, &root);
        let clone = Repo::at(&path);
        let verdicts: Vec<PinVerdict> = if !clone.is_repo() {
            vec![
                PinVerdict::Unverifiable(format!(
                    "aucun clone local en {} (--clone {}=<chemin>)",
                    path.display(),
                    repo.name
                ));
                repo.revs.len()
            ]
        } else {
            if o.fetch {
                if let Some(remote) = remote_of(&o.reference) {
                    if let Err(e) = clone.fetch(remote) {
                        eprintln!(
                            "pins : git fetch {remote} dans {} a échoué ({e}) ; verdict sur les refs locales",
                            path.display()
                        );
                    }
                }
            }
            repo.revs
                .iter()
                .map(|r| check_rev(&clone, &o.reference, &r.rev))
                .collect()
        };
        print!("{}", render_repo(repo, &verdicts));
        code = code.max(
            verdicts
                .iter()
                .map(PinVerdict::exit_code)
                .max()
                .unwrap_or(0),
        );
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testrepo::TempRepo;

    const MANIFEST: &str = r#"
[dependencies]
tokio = { workspace = true }
nexus-claude = { git = "https://github.com/this-rs/nexus.git", rev = "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904", features = ["memory"] }
# The path resolver of nexus-tools. Same pin as nexus-claude.
nexus-tools = { git = "https://github.com/this-rs/nexus.git", rev = "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904" }
other = { git = "https://example.org/acme/other", branch = "main" }

[dev-dependencies]
nexus-claude = { git = "https://github.com/this-rs/nexus.git", rev = "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904", features = ["testkit"] }

[dev-dependencies.widget]
git = "ssh://git@github.com/acme/widget.git"
rev = "0123456789abcdef0123456789abcdef01234567"
features = ["x"]

[profile.release]
opt-level = 3
"#;

    #[test]
    fn parse_pins_reads_inline_tables_and_dotted_tables() {
        let pins = parse_pins(MANIFEST);
        let as_tuples: Vec<(String, String, String)> = pins
            .iter()
            .map(|p| (p.dep.clone(), repo_name(&p.url), p.rev.clone()))
            .collect();
        let expect = |d: &str, r: &str, s: &str| (d.to_string(), r.to_string(), s.to_string());
        assert_eq!(
            as_tuples,
            vec![
                expect(
                    "nexus-claude",
                    "nexus",
                    "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904"
                ),
                expect(
                    "nexus-tools",
                    "nexus",
                    "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904"
                ),
                expect(
                    "nexus-claude",
                    "nexus",
                    "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904"
                ),
                expect(
                    "widget",
                    "widget",
                    "0123456789abcdef0123456789abcdef01234567"
                ),
            ]
        );
        assert_eq!(pins[0].line, 4);
        // A dotted table is located at its header line.
        assert_eq!(pins[3].line, 12);
    }

    #[test]
    fn group_by_repo_merges_the_same_rev_and_keeps_distinct_ones() {
        let mut pins = parse_pins(MANIFEST);
        pins.push(Pin {
            dep: "nexus-extra".into(),
            url: "https://github.com/this-rs/nexus.git".into(),
            rev: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef".into(),
            line: 99,
        });
        let repos = group_by_repo(&pins);
        assert_eq!(repos.len(), 2);
        assert_eq!(repos[0].name, "nexus");
        assert_eq!(repos[0].revs.len(), 2);
        assert_eq!(repos[0].revs[0].deps, vec!["nexus-claude", "nexus-tools"]);
        assert_eq!(repos[0].revs[1].deps, vec!["nexus-extra"]);
        assert_eq!(repos[1].name, "widget");
    }

    #[test]
    fn repo_name_strips_host_path_and_git_suffix() {
        assert_eq!(repo_name("https://github.com/this-rs/nexus.git"), "nexus");
        assert_eq!(repo_name("https://github.com/this-rs/nexus"), "nexus");
        assert_eq!(repo_name("git@github.com:acme/widget.git"), "widget");
    }

    #[test]
    fn toml_str_does_not_confuse_a_key_with_a_suffix_of_another() {
        assert_eq!(
            toml_str(r#"x = { git = "u", rev = "r" }"#, "rev").as_deref(),
            Some("r")
        );
        assert_eq!(toml_str(r#"x = { git = "u", brev = "r" }"#, "rev"), None);
        assert_eq!(toml_str(r#"x = { git = "u", rev = 3 }"#, "rev"), None);
    }

    #[test]
    fn parse_args_collects_clones() {
        let args: Vec<String> = ["--clone", "nexus=/tmp/n", "--no-fetch", "--ref", "main"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let o = parse_args(&args).unwrap();
        assert_eq!(
            o.clones,
            vec![("nexus".to_string(), PathBuf::from("/tmp/n"))]
        );
        assert!(!o.fetch);
        assert_eq!(o.reference, "main");
        assert!(parse_args(&["--clone".to_string(), "nexus".to_string()]).is_err());
    }

    #[test]
    fn clone_path_prefers_the_explicit_clone_over_the_sibling() {
        let root = Path::new("/w/backend");
        assert_eq!(
            clone_path("nexus", &[], root),
            PathBuf::from("/w/backend/../nexus")
        );
        let clones = vec![("nexus".to_string(), PathBuf::from("/elsewhere/nexus"))];
        assert_eq!(
            clone_path("nexus", &clones, root),
            PathBuf::from("/elsewhere/nexus")
        );
    }

    fn repo_with(rev: &str) -> RepoPins {
        RepoPins {
            name: "nexus".into(),
            url: "https://github.com/this-rs/nexus.git".into(),
            revs: vec![RevUse {
                rev: rev.into(),
                deps: vec!["nexus-claude".into(), "nexus-tools".into()],
            }],
        }
    }

    #[test]
    fn render_up_to_date_and_unverifiable() {
        let repo = repo_with("b3afe82fc576e80e7ffd7c39b0c25b8f90d72904");
        let text = render_repo(
            &repo,
            &[PinVerdict::UpToDate {
                main: "b3afe82fc576e80e7ffd7c39b0c25b8f90d72904".into(),
            }],
        );
        assert_eq!(text, "nexus main ↔ épingle b3afe82f (nexus-claude, nexus-tools) : à jour (main = b3afe82f)\n");
        let text = render_repo(
            &repo,
            &[PinVerdict::Unverifiable(
                "aucun clone local en ../nexus".into(),
            )],
        );
        assert_eq!(text, "nexus main ↔ épingle b3afe82f (nexus-claude, nexus-tools) : non vérifiable — aucun clone local en ../nexus\n");
    }

    #[test]
    fn render_behind_lists_the_subjects() {
        let repo = repo_with("b3afe82fc576e80e7ffd7c39b0c25b8f90d72904");
        let v = PinVerdict::Behind {
            main: "8373c2917908e66c9ac92fd1e1f046e9cc8263ba".into(),
            merges: vec![
                Commit {
                    sha: "8373c2917908".into(),
                    subject: "Fix interleaving (#128)".into(),
                },
                Commit {
                    sha: "d40f878a0000".into(),
                    subject: "docs(diagrams): N13 (#127)".into(),
                },
            ],
        };
        assert_eq!(
            render_repo(&repo, &[v]),
            "nexus main ↔ épingle b3afe82f (nexus-claude, nexus-tools) : 2 commits de retard sur main 8373c291 (sha de main)\n  - 8373c291 Fix interleaving (#128)\n  - d40f878a docs(diagrams): N13 (#127)\n"
        );
    }

    #[test]
    fn render_two_distinct_pins_announces_the_inconsistency_first() {
        let mut repo = repo_with("aaaaaaaaaaaa");
        repo.revs.push(RevUse {
            rev: "bbbbbbbbbbbb".into(),
            deps: vec!["nexus-extra".into()],
        });
        let text = render_repo(
            &repo,
            &[
                PinVerdict::UpToDate {
                    main: "aaaaaaaaaaaa".into(),
                },
                PinVerdict::Unverifiable("x".into()),
            ],
        );
        assert!(text.starts_with("nexus : 2 épingles différentes dans Cargo.toml (nexus-claude, nexus-tools = aaaaaaaa ; nexus-extra = bbbbbbbb)\n"), "{text}");
        assert_eq!(text.lines().count(), 3);
    }

    // Red/green: a deliberately stale pin is reported, an up-to-date one is not.
    #[test]
    fn a_stale_pin_is_behind_and_the_head_pin_is_up_to_date() {
        let nexus = TempRepo::new();
        let old = nexus.commit("chore: retrait du code mort (#125)");
        nexus.commit("docs(diagrams): relus (#127)");
        let head = nexus.commit("Fix concurrent fake processes (#128)");
        let clone = Repo::at(nexus.path());

        let stale = check_rev(&clone, "main", &old);
        let PinVerdict::Behind { main, merges } = &stale else {
            panic!("{stale:?}");
        };
        assert_eq!(main, &head);
        assert_eq!(
            merges
                .iter()
                .map(|m| m.subject.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Fix concurrent fake processes (#128)",
                "docs(diagrams): relus (#127)"
            ]
        );
        assert_eq!(stale.exit_code(), 1);
        let text = render_repo(&repo_with(&old), std::slice::from_ref(&stale));
        assert!(text.contains(": 2 commits de retard sur main"), "{text}");

        let fresh = check_rev(&clone, "main", &head);
        assert_eq!(fresh, PinVerdict::UpToDate { main: head.clone() });
        assert_eq!(fresh.exit_code(), 0);
        let text = render_repo(&repo_with(&head), &[fresh]);
        assert!(text.contains(": à jour"), "{text}");
        assert!(!text.contains("retard"), "{text}");
    }

    #[test]
    fn an_intermediate_sha_is_off_main_and_an_unknown_one_is_unverifiable() {
        let nexus = TempRepo::new();
        let base = nexus.commit("base (#1)");
        let branch = nexus.side_branch("feat/x", &base, "intermediate commit of a PR");
        nexus.commit("squash of that PR (#2)");
        let clone = Repo::at(nexus.path());

        let v = check_rev(&clone, "main", &branch);
        let PinVerdict::OffMain {
            merges,
            commits_off,
            ..
        } = &v
        else {
            panic!("{v:?}");
        };
        assert_eq!(*commits_off, 1);
        assert_eq!(merges.len(), 1);
        let text = render_repo(&repo_with(&branch), &[v]);
        assert!(text.contains("sha intermédiaire, hors de main"), "{text}");
        assert!(
            text.contains("1 commit que main n'a pas ; main a 1 commit que l'épingle n'a pas"),
            "{text}"
        );

        let v = check_rev(&clone, "main", "0123456789abcdef0123456789abcdef01234567");
        assert!(
            matches!(&v, PinVerdict::Unverifiable(why) if why.contains("inconnu du clone")),
            "{v:?}"
        );
        assert_eq!(v.exit_code(), 2);
        let v = check_rev(&clone, "origin/main", &base);
        assert!(
            matches!(&v, PinVerdict::Unverifiable(why) if why.contains("origin/main")),
            "{v:?}"
        );
    }
}
