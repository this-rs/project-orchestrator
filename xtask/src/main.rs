//! `cargo xtask` — repository automation for the backend.
//!
//! A small dispatcher: each subcommand lives in its own module, exposes a
//! `run(args) -> i32` entry point that is the ONLY place touching disk,
//! network or `git`, and keeps its logic in pure functions with unit tests.
//! A later task (P8.1, storage boundary) adds its own subcommand here.
//!
//! Exit codes, shared by every subcommand: 0 = nothing to report (identical,
//! up to date), 1 = a divergence was CONSTATED, 2 = could not verify (missing
//! clone, unreachable backend, bad arguments). None of them acts on anything.

mod git;
mod in_service;
mod pins;
#[cfg(test)]
mod testrepo;

const USAGE: &str = "\
usage: cargo xtask <subcommand> [options]

  in-service   is the running backend the current main? (AM-1)
  pins         did main of each git-pinned repo reach Cargo.toml? (AM-2)

`cargo xtask <subcommand> --help` lists the options of a subcommand.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("in-service") => in_service::run(&args[1..]),
        Some("pins") => pins::run(&args[1..]),
        Some("-h") | Some("--help") | Some("help") => {
            println!("{USAGE}");
            0
        }
        Some(other) => {
            eprintln!("xtask: sous-commande inconnue « {other} »\n\n{USAGE}");
            2
        }
        None => {
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}
