//! `cargo xtask containment` — does this branch contain the remote's main?
//!
//! The consolidation branch lives apart from `main`, but `main` is merged INTO
//! it every week (CONSOLIDATION.md, rule 2). Nothing enforced that, so the
//! branch could drift for months and only be found unfusable at the end. This
//! subcommand CONSTATE the drift; it never merges, never rebases, never writes.

use crate::git::{short, Repo};
use std::path::PathBuf;

const HELP: &str = "\
usage: cargo xtask containment [options]

  --remote <name>   remote to compare against, default origin
  --branch <name>   branch on that remote, default main
  --head <ref>      what « here » means, default HEAD
  --repo <path>     checkout to inspect, default the one containing the cwd
  --no-fetch        do not `git fetch` first (offline replay)
  --max-behind <n>  tolerate up to n commits of drift, default 3 (0 = strict)

tells whether every commit of <remote>/<branch> is reachable from <head>.

exit 0  the branch contains the remote branch, or trails it within --max-behind
exit 1  it trails by more than --max-behind — the drift is constated
exit 2  could not verify (no repo, unknown ref, fetch refused)";

/// How far this branch may trail the remote before the gate turns red.
///
/// Measured 11/11/2026: `main` took 4 commits in 12 h, and each one turned the
/// branch red for a condition GitHub ALREADY enforces (`strict: true` on main
/// refuses a stale branch at merge time). A gate that is red for a benign
/// reason is a gate everybody learns to ignore — hence a tolerance.
pub const DEFAULT_MAX_BEHIND: usize = 3;

/// `Ok(None)` when `head` contains `remote`; `Ok(Some(behind))` when it trails by
/// at most `max_behind` (tolerated, and still reported); `Err(behind)` beyond.
pub fn classify(contains: bool, behind: usize, max_behind: usize) -> Result<Option<usize>, usize> {
    if contains {
        return Ok(None);
    }
    let behind = behind.max(1);
    if behind <= max_behind {
        Ok(Some(behind))
    } else {
        Err(behind)
    }
}

fn value(args: &[String], i: &mut usize, flag: &str) -> Result<String, i32> {
    *i += 1;
    match args.get(*i) {
        Some(v) => Ok(v.clone()),
        None => {
            eprintln!("xtask containment: {flag} attend une valeur");
            Err(2)
        }
    }
}

pub fn run(args: &[String]) -> i32 {
    let mut remote = String::from("origin");
    let mut branch = String::from("main");
    let mut head = String::from("HEAD");
    let mut repo_path: Option<PathBuf> = None;
    let mut fetch = true;
    let mut max_behind = DEFAULT_MAX_BEHIND;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => {
                println!("{HELP}");
                return 0;
            }
            "--no-fetch" => fetch = false,
            "--max-behind" => match value(args, &mut i, "--max-behind") {
                Ok(v) => match v.parse::<usize>() {
                    Ok(n) => max_behind = n,
                    Err(_) => {
                        eprintln!(
                            "xtask containment: --max-behind attend un entier positif, reçu « {v} »"
                        );
                        return 2;
                    }
                },
                Err(c) => return c,
            },
            "--remote" => match value(args, &mut i, "--remote") {
                Ok(v) => remote = v,
                Err(c) => return c,
            },
            "--branch" => match value(args, &mut i, "--branch") {
                Ok(v) => branch = v,
                Err(c) => return c,
            },
            "--head" => match value(args, &mut i, "--head") {
                Ok(v) => head = v,
                Err(c) => return c,
            },
            "--repo" => match value(args, &mut i, "--repo") {
                Ok(v) => repo_path = Some(PathBuf::from(v)),
                Err(c) => return c,
            },
            other => {
                eprintln!("xtask containment: option inconnue « {other} »\n\n{HELP}");
                return 2;
            }
        }
        i += 1;
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let repo = match repo_path {
        Some(p) => Repo::at(p),
        None => match Repo::discover(&cwd) {
            Some(r) => r,
            None => {
                eprintln!(
                    "xtask containment: aucun dépôt git au-dessus de {}",
                    cwd.display()
                );
                return 2;
            }
        },
    };

    let target = format!("{remote}/{branch}");
    if fetch {
        if let Err(e) = repo.fetch(&remote) {
            eprintln!("xtask containment: `git fetch {remote}` a échoué : {e}");
            eprintln!(
                "  (réseau coupé ? relancer avec --no-fetch pour juger sur les refs locales)"
            );
            return 2;
        }
    }

    let head_sha = match repo.resolve(&head) {
        Some(s) => s,
        None => {
            eprintln!("xtask containment: « {head} » ne désigne aucun commit connu");
            return 2;
        }
    };
    let remote_sha = match repo.resolve(&target) {
        Some(s) => s,
        None => {
            eprintln!("xtask containment: « {target} » ne désigne aucun commit connu");
            eprintln!("  (remote absent, ou `git fetch {remote}` jamais fait)");
            return 2;
        }
    };

    let contains = repo.is_ancestor(&remote_sha, &head_sha);
    let behind = repo.count_between(&head_sha, &remote_sha).unwrap_or(0);

    match classify(contains, behind, max_behind) {
        Ok(None) => {
            println!(
                "containment: {} contient {target} ({})",
                short(&head_sha),
                short(&remote_sha)
            );
            0
        }
        Ok(Some(behind)) => {
            println!(
                "containment: {} ne contient pas {target} — {behind} commit(s) de retard, tolérés (max {max_behind})",
                short(&head_sha)
            );
            if let Ok(commits) = repo.merges_between(&head_sha, &remote_sha) {
                for c in commits.iter().take(max_behind) {
                    println!("    {} {}", short(&c.sha), c.subject);
                }
            }
            println!("  Merger avant la prochaine revue : `git merge {target}`.");
            0
        }
        Err(behind) => {
            println!("containment: CETTE BRANCHE NE CONTIENT PAS {target}");
            println!("  {behind} commit(s) de {target} absent(s) d'ici.");
            if let Ok(commits) = repo.merges_between(&head_sha, &remote_sha) {
                for c in commits.iter().take(10) {
                    println!("    {} {}", short(&c.sha), c.subject);
                }
                if commits.len() > 10 {
                    println!("    … et {} autre(s)", commits.len() - 10);
                }
            }
            println!();
            println!("À faire : `git merge {target}` DANS cette branche, puis relancer.");
            println!("Règle 2 de CONSOLIDATION.md : main redescend ici, jamais l'inverse.");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_when_the_remote_is_an_ancestor() {
        assert_eq!(classify(true, 0, 3), Ok(None));
        // A stale count must not turn a healthy branch red.
        assert_eq!(classify(true, 3, 3), Ok(None));
    }

    #[test]
    fn a_short_drift_is_tolerated_and_still_reported() {
        assert_eq!(classify(false, 1, 3), Ok(Some(1)));
        assert_eq!(classify(false, 3, 3), Ok(Some(3)));
    }

    #[test]
    fn beyond_the_tolerance_the_gate_turns_red() {
        assert_eq!(classify(false, 4, 3), Err(4));
        assert_eq!(classify(false, 40, 3), Err(40));
    }

    #[test]
    fn strict_mode_is_the_old_behaviour() {
        // --max-behind 0 : any drift is red, exactly as before the tolerance.
        assert_eq!(classify(false, 1, 0), Err(1));
    }

    #[test]
    fn an_unreadable_count_still_means_behind() {
        // Never report zero drift: classify(false, 0) is at least one commit.
        assert_eq!(classify(false, 0, 3), Ok(Some(1)));
        assert_eq!(classify(false, 0, 0), Err(1));
    }
}
