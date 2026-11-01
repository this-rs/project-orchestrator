# Operations: after-merge checks

Two questions nobody could answer on 15/10 without reading three terminals:
a fix was merged while the running backend predated it, and a nexus merge had
not reached the backend's pin. `cargo xtask` answers both with one sentence
each. The checks **constate only**: no restart, no deploy, no write to any
repository. The only network they use is a read-only `GET /api/version` on
the backend and an optional `git fetch`.

They live in `xtask/`, a std-only crate outside the workspace, wired through
the `xtask` alias of `.cargo/config.toml`; run them from the repo root.

```bash
cargo xtask in-service   # AM-1: is what runs main?
cargo xtask pins         # AM-2: did main of each git-pinned repo reach Cargo.toml?
cargo xtask <subcommand> --help
```

Exit codes, shared by both: `0` nothing to report, `1` a divergence was
constated, `2` could not verify (the reason is on stderr and names what is
missing).

## `in-service` — is what runs main?

Reads the running revision from `GET <url>/api/version`: `build.git_sha`
(the full sha baked by `build.rs`), or, for a backend built before that field
existed, the abbreviated sha inside `update.current_build`
(`0.0.17+353 (c94fa37d)`). Resolves it in the local checkout and compares it
with `origin/main` (after a `git fetch origin`, unless `--no-fetch`).

```text
$ cargo xtask in-service
in-service : en service lu depuis update.current_build (c94fa37d)
main = 0ab7a21f, en service = c94fa37d, identiques : non, écart : 6 fusions
  - 0ab7a21f ci: hash only the pinned toolchain into the rust-cache keys (#895)
  - 1a2b3c4d fix(chat): … (#894)
  …
```

« écart » counts the first-parent commits of main the service lacks, i.e. the
merges (squash or merge commit). A service built from a branch is reported as
« en service n'est pas sur origin/main : N commits hors main ». A sha the local
clone does not know gives « écart : inconnu » and exit 2 (fetch, or the
service runs a commit that was never pushed).

| Option | Meaning |
|---|---|
| `--url <base>` | backend to read, default `http://localhost:8080` |
| `--ref <ref>` | what « main » means, default `origin/main` |
| `--repo <path>` | backend checkout to compare in, default: the one containing the cwd |
| `--no-fetch` | do not fetch first (offline replay) |
| `--service-sha <sha>` | replay: take this as the running revision, no HTTP |
| `--version-json <file>` | replay: read a saved `/api/version` body, no HTTP |

Replay a past incident offline: save the response once
(`curl -s http://localhost:8080/api/version > /tmp/v.json`), then
`cargo xtask in-service --no-fetch --version-json /tmp/v.json`.

## `pins` — did a merge in one repo reach the other?

Reads every `git = "…", rev = "…"` dependency of `Cargo.toml` (inline tables
and `[dependencies.name]` tables, `[patch]` included), groups them by
repository, and checks each distinct rev in a **local clone** of that
repository: `--clone <repo>=<path>`, else the sibling directory `../<repo>` of
the backend checkout (on this machine: `backend/` and `nexus/` side by side).
From a linked worktree (`_wt/<name>/`) the sibling is `_wt/nexus`, which does
not exist: pass `--clone nexus=/path/to/nexus` there. The clone is fetched
first unless `--no-fetch`.

```text
$ cargo xtask pins
nexus main ↔ épingle b3afe82f (nexus-claude, nexus-tools) : 2 commits de retard sur main 8373c291 (sha de main)
  - 8373c291 Fix concurrent fake processes interleaving record writes (#128)
  - d40f878a docs(diagrams): nexus-sdk-* relus contre le code (N13) (#127)
```

Four verdicts per repository:

- « à jour (main = …) » — the pin is the head of main;
- « N commits de retard sur main … (sha de main) » — the pin is on main, these
  merges came after it;
- « sha intermédiaire, hors de main … » — main does not contain the pin (a
  branch or a commit later squashed);
- « non vérifiable — … » — no local clone, a sha the clone does not know, or a
  missing `origin/main`: the repo is **named**, never skipped.

When one repository is pinned at two different revs (e.g. `nexus-claude` and
`nexus-tools` drifting apart), the first line says so and each rev gets its
own verdict.

| Option | Meaning |
|---|---|
| `--manifest <file>` | Cargo.toml to read, default: the one at the repo root |
| `--repo <path>` | backend checkout, default: the one containing the cwd |
| `--clone <name>=<path>` | local clone of a pinned repo (repeatable) |
| `--ref <ref>` | what « main » means in each clone, default `origin/main` |
| `--no-fetch` | do not fetch the clones first (offline replay) |

## Tests

`cargo test --manifest-path xtask/Cargo.toml`. The logic is in pure
functions tested from captured `git log` / `rev-list` / `/api/version`
strings; the red/green cases build throwaway git repositories under the
system temp directory: two simulated trees (identical and divergent) for
`in-service`, a deliberately stale pin versus the head pin for `pins`. Only
`run()` touches the real disk and network. CI (lint job) formats, lints and
tests the crate, then runs `in-service` HEAD-against-HEAD as a smoke test.

## Starting a copy of a production database

Before an upgrade that carries data migrations (`neo4j/data_migrations.rs`,
for instance `2026-10-backfill-stable-key` and `2026-10-backfill-file-uid`),
run the new binary on a **restored copy** of the production database, not on
production. Each point below was met in practice.

1. **Back up consistently.** Neo4j Community cannot dump a running database:
   stop it, run `neo4j-admin database dump neo4j --to-path=<dir>`, start it
   again. Meilisearch: `POST /dumps`.
2. **Restore into separate containers**, on other ports and new volumes,
   never into the production ones. Give the throwaway Meilisearch a master
   key: a keyless instance rejects the empty key of the config and the
   server stops at boot with `invalid_api_key`.
3. **Pass an explicit `--config`** that points at the copy. Auto-detection
   finds the production `config.yaml`.
4. **Isolate `HOME`.** The server also writes under `~/.claude*`, the vault
   files and `~/.project-orchestrator`, and the `claude` CLI looks its
   transcripts up there.
5. **Set `RESUME_ON_START=false`.** The copy still holds the sessions that
   were live when it was taken. Without the switch the server launches the
   `claude` CLI in the real working directory to resume them, relaunches the
   interrupted plan runs, and starts the schedule and event triggers. With
   it, none of that happens; the boot log says so (`RESUME_ON_START=false:
   no session or plan run is relaunched at boot`).
6. **Mind the listener.** With no `auth` section the server answers every
   request anonymously on `0.0.0.0` and logs `AUTH DISABLED` at boot. Stop it
   as soon as the check is done, or configure `auth`.
7. **A run is not read-only.** `RESUME_ON_START=false` keeps the agents
   asleep, not the maintenance: the file watcher resyncs the projects whose
   path exists on the machine, and the heartbeat prunes weak synapses
   (`DecaySynapses`). Reload the copy from the dump to start over.

Data migrations are resumable. Each one commits in batches and is recorded by
a `(:DataMigration {id})` marker once complete; a restart after a cut skips
the nodes already written and a restart after completion replays nothing. To
check a run, compare the marker count with the number of migrations the
binary declares, and count the nodes still without their new property.
