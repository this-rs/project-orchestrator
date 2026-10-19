//! `cargo xtask in-service` — is what runs main? (AM-1)
//!
//! Prints ONE sentence, « main = <sha>, en service = <sha>, identiques :
//! oui/non, écart : N fusions », and, when the two differ, the merges (first-
//! parent commits of main) the running backend lacks. It CONSTATES only: no
//! restart, no deploy, no write anywhere. The only network calls are a
//! read-only `GET /api/version` on the backend and an optional `git fetch`.
//!
//! The running revision comes from `GET /api/version`: `build.git_sha`
//! (full sha, baked by build.rs — added with this subcommand) or, for a
//! backend built before that field existed, the abbreviated sha inside
//! `update.current_build` (`0.0.17+353 (c94fa37d)`, from `git describe`).

use crate::git::{looks_like_sha, remote_of, short, Commit, Repo};
use std::path::{Path, PathBuf};
use std::process::Command;

const HELP: &str = "\
usage: cargo xtask in-service [options]

  --url <base>          backend to read (GET <base>/api/version), default http://localhost:8080
  --ref <ref>           what « main » means, default origin/main
  --repo <path>         backend checkout to compare in, default: the one containing the cwd
  --no-fetch            do not `git fetch` the remote of --ref first (offline replay)
  --service-sha <sha>   replay: take this as the running revision, no HTTP
  --version-json <file> replay: read a saved /api/version response, no HTTP

exit 0 identical, 1 the service is behind main, 2 could not verify.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub url: String,
    pub reference: String,
    pub repo: Option<PathBuf>,
    pub fetch: bool,
    pub service_sha: Option<String>,
    pub version_json: Option<PathBuf>,
    pub help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            url: "http://localhost:8080".to_string(),
            reference: "origin/main".to_string(),
            repo: None,
            fetch: true,
            service_sha: None,
            version_json: None,
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
            "--url" => o.url = value("--url")?.trim_end_matches('/').to_string(),
            "--ref" => o.reference = value("--ref")?,
            "--repo" => o.repo = Some(PathBuf::from(value("--repo")?)),
            "--no-fetch" => o.fetch = false,
            "--service-sha" => o.service_sha = Some(value("--service-sha")?),
            "--version-json" => o.version_json = Some(PathBuf::from(value("--version-json")?)),
            "-h" | "--help" => o.help = true,
            other => return Err(format!("option inconnue « {other} »")),
        }
    }
    Ok(o)
}

/// Where the running revision was read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceRevision {
    pub sha: String,
    pub source: &'static str,
}

/// Value of the first `"key": "<string>"` in a JSON document, without a JSON
/// parser: enough for the two flat string fields this check reads, and
/// std-only. Returns `None` for a missing key or a non-string value.
fn json_string_field(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut from = 0;
    while let Some(pos) = json[from..].find(&needle) {
        let after = &json[from + pos + needle.len()..];
        let after = after.trim_start();
        if let Some(rest) = after.strip_prefix(':') {
            let rest = rest.trim_start();
            if let Some(s) = rest.strip_prefix('"') {
                let end = s.find('"')?;
                return Some(s[..end].to_string());
            }
            return None;
        }
        from += pos + needle.len();
    }
    None
}

/// The running revision from a `GET /api/version` body: `build.git_sha` when
/// the backend exposes it, else the `(<sha>)` of `update.current_build`.
pub fn service_revision(version_json: &str) -> Option<ServiceRevision> {
    if let Some(sha) = json_string_field(version_json, "git_sha") {
        if looks_like_sha(&sha) {
            return Some(ServiceRevision {
                sha,
                source: "build.git_sha",
            });
        }
    }
    let build = json_string_field(version_json, "current_build")?;
    let open = build.rfind('(')?;
    let close = build[open..].find(')')? + open;
    let sha = build[open + 1..close].trim();
    looks_like_sha(sha).then(|| ServiceRevision {
        sha: sha.to_string(),
        source: "update.current_build",
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceState {
    /// The local repo does not know this sha (stale clone, or not a sha).
    Unknown,
    Known {
        sha: String,
        on_main: bool,
        /// Merges of main the service lacks (first-parent, newest first).
        merges_behind: Vec<Commit>,
        /// Commits the service has that main does not (0 when on main).
        commits_off_main: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    pub main_ref: String,
    pub main: String,
    /// The revision as the service reported it (possibly abbreviated).
    pub service: String,
    pub state: ServiceState,
}

impl Comparison {
    pub fn identical(&self) -> bool {
        matches!(&self.state, ServiceState::Known { sha, on_main: true, merges_behind, .. }
            if merges_behind.is_empty() && *sha == self.main)
    }

    pub fn exit_code(&self) -> i32 {
        match &self.state {
            ServiceState::Unknown => 2,
            ServiceState::Known { .. } if self.identical() => 0,
            ServiceState::Known { .. } => 1,
        }
    }
}

/// Compare `service_sha` with `main_ref` in `repo` (git reads only).
pub fn compare(repo: &Repo, main_ref: &str, service_sha: &str) -> Result<Comparison, String> {
    let main = repo.resolve(main_ref).ok_or_else(|| {
        format!(
            "« {main_ref} » n'est pas un commit de {}",
            repo.path.display()
        )
    })?;
    let state = match repo.resolve(service_sha) {
        None => ServiceState::Unknown,
        Some(sha) => {
            let on_main = repo.is_ancestor(&sha, &main);
            let merges_behind = repo.merges_between(&sha, &main)?;
            let commits_off_main = if on_main {
                0
            } else {
                repo.count_between(&main, &sha)?
            };
            ServiceState::Known {
                sha,
                on_main,
                merges_behind,
                commits_off_main,
            }
        }
    };
    Ok(Comparison {
        main_ref: main_ref.to_string(),
        main,
        service: service_sha.to_string(),
        state,
    })
}

/// French agreement: 0 and 1 take the singular (« 0 fusion », « 1 commit »).
fn plural(n: usize, one: &str, many: &str) -> String {
    if n <= 1 {
        format!("{n} {one}")
    } else {
        format!("{n} {many}")
    }
}

/// The sentence (and the merge list under it when they differ).
pub fn render(c: &Comparison) -> String {
    let main = short(&c.main);
    match &c.state {
        ServiceState::Unknown => format!(
            "main = {main}, en service = {}, identiques : non, écart : inconnu (sha inconnu du dépôt local : git fetch ?)",
            c.service
        ),
        ServiceState::Known {
            sha,
            on_main,
            merges_behind,
            commits_off_main,
        } => {
            let identical = c.identical();
            let mut out = format!(
                "main = {main}, en service = {}, identiques : {}, écart : {}",
                short(sha),
                if identical { "oui" } else { "non" },
                plural(merges_behind.len(), "fusion", "fusions"),
            );
            if !on_main {
                out.push_str(&format!(
                    " (en service n'est pas sur {} : {} hors main)",
                    c.main_ref,
                    plural(*commits_off_main, "commit", "commits")
                ));
            }
            for m in merges_behind {
                out.push_str(&format!("\n  - {} {}", short(&m.sha), m.subject));
            }
            out
        }
    }
}

fn read_version(o: &Options) -> Result<ServiceRevision, String> {
    if let Some(sha) = &o.service_sha {
        return Ok(ServiceRevision {
            sha: sha.clone(),
            source: "--service-sha",
        });
    }
    let body = match &o.version_json {
        Some(file) => {
            std::fs::read_to_string(file).map_err(|e| format!("{} : {e}", file.display()))?
        }
        None => http_get(&format!("{}/api/version", o.url))?,
    };
    service_revision(&body).ok_or_else(|| {
        "/api/version ne porte ni build.git_sha ni update.current_build « (<sha>) » : \
             le backend en service est-il antérieur à ce champ et sans service de mise à jour ?"
            .to_string()
    })
}

/// Read-only GET through `curl` (https and proxies for free, nothing beyond std here).
fn http_get(url: &str) -> Result<String, String> {
    let out = Command::new("curl")
        .args(["-sS", "--fail", "--max-time", "10", url])
        .output()
        .map_err(|e| format!("curl introuvable : {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(format!(
            "GET {url} : {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
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
            eprintln!("in-service : {e}\n\n{HELP}");
            return 2;
        }
    };
    if o.help {
        println!("{HELP}");
        return 0;
    }
    let outcome = (|| -> Result<Comparison, String> {
        let repo = locate_repo(&o)?;
        if o.fetch {
            if let Some(remote) = remote_of(&o.reference) {
                if let Err(e) = repo.fetch(remote) {
                    eprintln!("in-service : git fetch {remote} a échoué ({e}) ; verdict sur les refs locales");
                }
            }
        }
        let rev = read_version(&o)?;
        eprintln!(
            "in-service : en service lu depuis {} ({})",
            rev.source, rev.sha
        );
        compare(&repo, &o.reference, &rev.sha)
    })();
    match outcome {
        Ok(c) => {
            println!("{}", render(&c));
            c.exit_code()
        }
        Err(e) => {
            eprintln!("in-service : non vérifiable — {e}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testrepo::TempRepo;

    const TODAY_RESPONSE: &str = r#"{"version":"0.1.0","features":{"embedded_frontend":false,"serve_frontend":true,"refs_v1":true},"build":{"target":"aarch64-macos","profile":"release"},"update":{"current":"0.0.17","current_build":"0.0.17+353 (c94fa37d)","latest":"0.0.17","update_available":false}}"#;

    #[test]
    fn service_revision_prefers_the_full_git_sha_field() {
        let json = r#"{"version":"0.1.0","build":{"target":"aarch64-macos","profile":"release","git_sha":"0ab7a21f1111111111111111111111111111aaaa"},"update":{"current_build":"0.0.17+359 (0ab7a21f)"}}"#;
        let rev = service_revision(json).unwrap();
        assert_eq!(rev.sha, "0ab7a21f1111111111111111111111111111aaaa");
        assert_eq!(rev.source, "build.git_sha");
    }

    #[test]
    fn service_revision_falls_back_to_the_describe_sha_of_an_older_backend() {
        let rev = service_revision(TODAY_RESPONSE).unwrap();
        assert_eq!(rev.sha, "c94fa37d");
        assert_eq!(rev.source, "update.current_build");
    }

    #[test]
    fn service_revision_ignores_an_empty_git_sha_and_a_release_build() {
        // Docker build: no git at build time, empty sha, and a release
        // `git describe` with no abbreviated commit in parentheses.
        let json = r#"{"build":{"git_sha":""},"update":{"current_build":"0.0.17"}}"#;
        assert_eq!(service_revision(json), None);
        let json = r#"{"build":{"git_sha":null},"update":{"current_build":"0.0.17+3 (c94fa37d)"}}"#;
        assert_eq!(service_revision(json).unwrap().sha, "c94fa37d");
        assert_eq!(service_revision("{}"), None);
    }

    #[test]
    fn json_string_field_skips_a_key_that_is_only_a_value() {
        let json = r#"{"note":"git_sha","build":{"git_sha":"abcdef0123"}}"#;
        assert_eq!(
            json_string_field(json, "git_sha").as_deref(),
            Some("abcdef0123")
        );
        assert_eq!(json_string_field(r#"{"n": 3}"#, "n"), None);
        assert_eq!(json_string_field(r#"{"a":"b"}"#, "zz"), None);
    }

    #[test]
    fn parse_args_reads_every_option_and_refuses_unknown_ones() {
        let args: Vec<String> = [
            "--url",
            "http://h:1/",
            "--ref",
            "upstream/main",
            "--no-fetch",
            "--service-sha",
            "abc1234",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let o = parse_args(&args).unwrap();
        assert_eq!(o.url, "http://h:1");
        assert_eq!(o.reference, "upstream/main");
        assert!(!o.fetch);
        assert_eq!(o.service_sha.as_deref(), Some("abc1234"));
        assert!(parse_args(&["--nope".to_string()]).is_err());
        assert!(parse_args(&["--url".to_string()]).is_err());
    }

    fn known(
        main: &str,
        sha: &str,
        on_main: bool,
        behind: &[(&str, &str)],
        off: usize,
    ) -> Comparison {
        Comparison {
            main_ref: "origin/main".into(),
            main: main.into(),
            service: short(sha).into(),
            state: ServiceState::Known {
                sha: sha.into(),
                on_main,
                merges_behind: behind
                    .iter()
                    .map(|(s, j)| Commit {
                        sha: s.to_string(),
                        subject: j.to_string(),
                    })
                    .collect(),
                commits_off_main: off,
            },
        }
    }

    #[test]
    fn render_identical() {
        let c = known(
            "0ab7a21f0000000000000000000000000000beef",
            "0ab7a21f0000000000000000000000000000beef",
            true,
            &[],
            0,
        );
        assert_eq!(
            render(&c),
            "main = 0ab7a21f, en service = 0ab7a21f, identiques : oui, écart : 0 fusion"
        );
        assert_eq!(c.exit_code(), 0);
    }

    #[test]
    fn render_behind_lists_the_merges() {
        let c = known(
            "0ab7a21f0000000000000000000000000000beef",
            "c94fa37d0000000000000000000000000000beef",
            true,
            &[
                ("0ab7a21f00", "ci: hash only the toolchain (#895)"),
                ("860ed17a00", "docs(diagrams): restore (#124)"),
            ],
            0,
        );
        assert_eq!(
            render(&c),
            "main = 0ab7a21f, en service = c94fa37d, identiques : non, écart : 2 fusions\n  - 0ab7a21f ci: hash only the toolchain (#895)\n  - 860ed17a docs(diagrams): restore (#124)"
        );
        assert_eq!(c.exit_code(), 1);
    }

    #[test]
    fn render_off_main_says_so() {
        let c = known(
            "aaaaaaaa00",
            "bbbbbbbb00",
            false,
            &[("aaaaaaaa00", "m (#1)")],
            3,
        );
        assert_eq!(
            render(&c),
            "main = aaaaaaaa, en service = bbbbbbbb, identiques : non, écart : 1 fusion (en service n'est pas sur origin/main : 3 commits hors main)\n  - aaaaaaaa m (#1)"
        );
        assert_eq!(c.exit_code(), 1);
    }

    #[test]
    fn render_unknown_is_unverifiable() {
        let c = Comparison {
            main_ref: "origin/main".into(),
            main: "aaaaaaaa00".into(),
            service: "deadbeef".into(),
            state: ServiceState::Unknown,
        };
        assert!(render(&c).contains("écart : inconnu"));
        assert_eq!(c.exit_code(), 2);
    }

    // Red/green on two simulated trees: the same binary revision against a
    // main that did not move (green) and one that took two merges (red).
    #[test]
    fn two_simulated_trees_give_two_distinct_verdicts() {
        let identical = TempRepo::new();
        let deployed = identical.commit("feat: the build in service (#1)");
        let same = compare(&Repo::at(identical.path()), "main", &deployed[..8]).unwrap();
        assert!(same.identical(), "{same:?}");
        assert_eq!(same.exit_code(), 0);
        assert_eq!(
            render(&same),
            format!(
                "main = {0}, en service = {0}, identiques : oui, écart : 0 fusion",
                &deployed[..8]
            )
        );

        let divergent = TempRepo::new();
        let deployed = divergent.commit("feat: the build in service (#1)");
        divergent.commit("fix: merged after the deploy (#2)");
        divergent.commit("feat: merged later still (#3)");
        let diff = compare(&Repo::at(divergent.path()), "main", &deployed[..8]).unwrap();
        assert!(!diff.identical());
        assert_eq!(diff.exit_code(), 1);
        let text = render(&diff);
        assert!(
            text.contains("identiques : non, écart : 2 fusions"),
            "{text}"
        );
        assert!(text.contains("fix: merged after the deploy (#2)"), "{text}");
        assert!(text.contains("feat: merged later still (#3)"), "{text}");
        assert!(
            text.lines().nth(1).unwrap().contains("(#3)"),
            "newest first: {text}"
        );
    }

    #[test]
    fn a_service_built_from_a_branch_is_reported_off_main() {
        let repo = TempRepo::new();
        let base = repo.commit("base (#1)");
        let branch = repo.side_branch("feat/x", &base, "wip on a branch");
        repo.commit("merged on main (#2)");
        let c = compare(&Repo::at(repo.path()), "main", &branch).unwrap();
        assert_eq!(c.exit_code(), 1);
        let text = render(&c);
        assert!(
            text.contains("n'est pas sur main : 1 commit hors main"),
            "{text}"
        );
        assert!(text.contains("écart : 1 fusion"), "{text}");
    }

    #[test]
    fn a_sha_the_clone_does_not_know_is_unverifiable() {
        let repo = TempRepo::new();
        repo.commit("only commit");
        let c = compare(
            &Repo::at(repo.path()),
            "main",
            "0123456789abcdef0123456789abcdef01234567",
        )
        .unwrap();
        assert_eq!(c.state, ServiceState::Unknown);
        assert_eq!(c.exit_code(), 2);
        assert!(compare(&Repo::at(repo.path()), "origin/main", "abc").is_err());
    }
}
