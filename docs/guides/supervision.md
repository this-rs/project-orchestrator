# Supervising the server (macOS launchd, Linux systemd)

A supervised server has two failure modes that look identical from the front
door and are not. This page names them, states the contract the binary owes the
supervisor, and gives the three commands that tell them apart.

## The contract: one owner per port, and a duplicate exits 0

The server binds the port **before** any initialization with side effects, so a
duplicate fails immediately instead of connecting to Neo4j, starting watchers and
recovering plan runs it does not own.

That is only half the contract. The other half is the **exit code**:

| Situation | Exit code | Why |
|---|---|---|
| The port is served by a **live orchestrator** (`/api/version` answers with `version` + `build`) | **0** | The supervisor's `KeepAlive { SuccessfulExit = false }` / `Restart=on-failure` relaunches on any *failure* code. Exiting 0 stops the loop by construction; the live instance keeps serving. |
| The port is held by **anything else** (a foreign process, a silent listener) | 1 | A real problem an operator must see. It is not masked. |
| Any other startup failure | 1 | Unchanged. |

The duplicate's message names the version it found and what to do; it is written
to the app log (`tracing`) **and** to the supervisor's redirect.

Why this matters, measured 2026-11-25: a hand-started server held `:8080` while
launchd's job relaunched the duplicate every 5 s — `runs = 590` in one day, 436
`orchestrator starting version` lines in the app log, **zero** errors in that
same log because the error goes to the supervisor's redirect. The mechanism is
`src/single_instance.rs` (probe + typed error) and `src/main.rs` (the exit-code
mapping); the binding site is `src/lib.rs`, search for "Bind the port FIRST".

## The redirect log is trimmed, because launchd never rotates it

`StandardOutPath` / `StandardErrorPath` point at `/tmp/orchestrator.log`. Nothing
rotates that file, and a crash loop writes a full boot per attempt — 1.0 GB in
one night. `ProgramArguments` therefore goes through
`scripts/launchd/po-supervise.sh`, which trims the file (default: keep the last
5 MB, `PO_SUPERVISOR_LOG` / `PO_SUPERVISOR_LOG_KEEP_BYTES` override) and then
`exec`s the server. The trim uses `cat > file`, never `mv`: launchd keeps the
file open in append mode, and a rename would send every later write to an
unlinked inode.

Reference copies of both live in `scripts/launchd/`; the installed files are
`~/Library/LaunchAgents/land.ffs.orchestrator.plist` and the wrapper next to it.

## Diagnostics: three commands

```bash
launchctl print gui/$(id -u)/land.ffs.orchestrator | grep -E "state|runs|last exit code|pid"
lsof -nP -iTCP:8080 -sTCP:LISTEN
ps -o pid,ppid,lstart,etime -p <pid>            # parent 1 = launchd, a shell = hand-started
tail -40 /tmp/orchestrator.log                   # where the real error lives
```

Read them in this order. `runs` climbing every few seconds while `pid` stays
empty means the job cannot start (port taken → verify with `lsof`). A `pid` whose
parent is a shell means the supervised instance is the one that cannot start, and
the shell-owned one is what serves the UI.

## Restarting, correctly

```bash
launchctl kickstart -k gui/$(id -u)/land.ffs.orchestrator
```

Never `./target/release/orchestrator serve` by hand: that instance owns the port
and the supervised one can never take it, which is exactly how the 590-relaunch
loop starts. If a hand-started instance is already running, stop it first.
