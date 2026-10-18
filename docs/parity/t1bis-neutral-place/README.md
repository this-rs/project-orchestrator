# T1-bis: the real Claude Code CLI in the host's neutral directory

The measurement behind the neutral place of a chat session (`src/chat/neutral_place.rs`, PO task
T1-bis 6dae6085, plan « Chat sans dépendance au cwd »). T1 (#608) tested the neutral directory with
a fake provider only; this replays by hand what the host does with the real CLI. Run it again **on
each Claude Code CLI bump** (`docs/parity/engines.md`, "À re-mesurer").

`./measure.sh <out-dir> [<neutral root>]`: makes `<root>/<uuid>` (0700, empty) at the host's real
root, runs one `claude -p` turn there on `claude-haiku-5-5` (no MCP server, no tool) asking it to
remember a random word, removes the directory and makes it again (what closing then resuming the
session does: `remove_for`, then `ensure`), then `claude -p --resume <cli session>` from the same
path asks for the word. It also lists any `CLAUDE.md`, `CLAUDE.local.md` or `.claude/` in the
ancestors of the root. About 0.01 USD.

## Result on Claude Code 2.1.296 (18/10/2026, 19:57 UTC, macOS)

| What | Measured |
|---|---|
| Turn 1 in `~/Library/Application Support/project-orchestrator/chat-neutral/<uuid>` | `success`, 0.0031 USD; the CLI wrote **nothing** in the directory (still empty after the turn) |
| Its transcript | `~/.claude/projects/-Users-<user>-Library-Application-Support-project-orchestrator-chat-neutral-<uuid>/<cli session>.jsonl` (keyed on the neutral path, outside the directory itself) |
| Directory removed, then made again at the same path, then `--resume` | `success`, 0.0055 USD, same CLI session id, **the word is recalled**: the conversation survives the removal of the directory |
| Control: `--resume` of the same CLI session from ANOTHER (fresh) neutral directory | also `success`, word recalled (0.0060 USD): on 2.1.296 the resume does not depend on the cwd. The deterministic path stays useful (the transcript stays under one project key, and an older or later CLI may look only there) |
| Ancestors of the root | none hold a `CLAUDE.md`, `CLAUDE.local.md` or `.claude/` except `~/.claude`, the user's own configuration that every session reads wherever it runs |

Total of the three turns: 0.0146 USD.

## What the measurement does not cover

- A `CLAUDE.md` or `.claude/` that a user puts in an ancestor of `<config dir>` (for example in
  `~/Library/` or `~/.config/`) WOULD be read by the CLI: the host does not guard against it, and
  the measurement only says there is none on this machine.
- Files the model writes in the neutral directory are lost when the session is closed (the
  directory is removed); the transcript in `~/.claude/projects/` is not.
- The host spawns the CLI with stream-json input and its own MCP server; this record uses `-p` with
  no MCP server. The resume path (`--resume <id>` from the stored `cli_session_id`) is the same.

## What would change the record

- the resume losing the conversation after the directory was made again: `neutral_place` must then
  keep the directory between turns (no removal at close) or the host must stop removing it;
- the CLI writing configuration in the neutral directory on its first turn: the directory is no
  longer "empty, no `.claude/`" as `neutral_place.rs` says.
