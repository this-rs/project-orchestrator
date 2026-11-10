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

tells whether every commit of <remote>/<branch> is reachable from <head>.

exit 0  the branch contains the remote branch
exit 1  it does not — the drift is constated, and the message says what to run
exit 2  could not verify (no repo, unknown ref, fetch refused)";

/// `Ok` when `head` contains `remote`, `Err(behind)` when it does not.
pub fn classify(contains: bool, behind: usize) -> Result<(), usize> {
    if contains {
        Ok(())
    } else {
        Err(behind.max(1))
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

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => {
                println!("{HELP}");
                return 0;
            }
            "--no-fetch" => fetch = false,
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

    match classify(contains, behind) {
        Ok(()) => {
            println!(
                "containment: {} contient {target} ({})",
                short(&head_sha),
                short(&remote_sha)
            );
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
        assert_eq!(classify(true, 0), Ok(()));
        // A stale count must not turn a healthy branch red.
        assert_eq!(classify(true, 3), Ok(()));
    }

    #[test]
    fn diverges_and_reports_a_count_of_at_least_one() {
        assert_eq!(classify(false, 4), Err(4));
        // An unreadable count still means "behind": never report zero drift.
        assert_eq!(classify(false, 0), Err(1));
    }
}
