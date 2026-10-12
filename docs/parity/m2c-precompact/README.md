# M2c: what a PreCompact hook answer does on the real Claude Code CLI

The measurement behind task M2c (PO task 79cc48df, note 172c164e). It needs an installed,
logged-in `claude`. No test runs it: the parity scenario replays the CLI with `fake_claude`,
so this has to be run again by hand **on each Claude Code CLI bump**
(`docs/parity/engines.md`, "À re-mesurer").

Every run uses `--model haiku`, `--setting-sources project`, a throwaway project directory,
and the instruction `End the summary with the exact line: ZEPHYR-42`. A run is positive when
the compact summary (the `isCompactSummary` user message of the stream-json output) contains
`ZEPHYR-42`.

| Script | What it measures |
|---|---|
| `sdkhost.mjs` | **The path nexus and the legacy engine use**: an SDK callback hook (`initialize` with `hooks.PreCompact`, then `hook_callback`). `node sdkhost.mjs <mode> <log-dir> <project-dir>`, mode `reason`, `addctx`, `sysmsg` or `empty`. |
| `command-hook.sh` + `precompact.sh` | A command hook from a settings file. `./command-hook.sh <out-dir> <mode>`, mode `reason`, `addctx`, `plain` or `none`. |

## Result on Claude Code 2.1.295 (12/10/2026)

SDK callback hook:

| Answer | Result |
|---|---|
| `{continue: true, reason: I}` (what the legacy engine sent before M2c) | ignored |
| `{hookSpecificOutput: {hookEventName: "PreCompact", additionalContext: I}}` | rejected (no PreCompact variant: "Error in hook callback") |
| `{systemMessage: I}` | **appended to the summary's instructions**, and also shown to the user in the "Compacted…" line |
| `{}` | nothing |

Command hook: plain stdout is appended to the instructions; JSON stdout is appended as raw
text, not parsed.

Both engines therefore answer PreCompact with `systemMessage` (legacy `CompactionNotifier`,
and the nexus Claude Code façade since nexus #94).
