# GROS-1b: what the real Claude Code CLI does with a tool result of about a megabyte

The measurement behind the row `tool_result.huge` of `docs/parity/engines.md` (PO task GROS-1b,
4b687177). The parity scenario replays the CLI with `fake_claude`, which runs no tool: what the
CLI does with the megabyte is measured here, by hand, and the scenario replays it
(`CLI_BIG_NOTICE` in `src/chat/engine_parity_tests.rs`). Run it again **on each Claude Code CLI
bump** (`docs/parity/engines.md`, "À re-mesurer").

`./measure.sh <fake_mcp> <out-dir>`: `claude -p` on `claude-haiku-5-5`, the nexus `fake_mcp` as
the only MCP server (`--strict-mcp-config`), its tool `big` allowed, one call with
`n = 1000000`, stream-json output. About 0.005 USD.

## Result on Claude Code 2.1.296 (18/10/2026)

Two runs (17:11 and 17:21 UTC), the same on both:

| What | Measured |
|---|---|
| What the model received for the call | a **text of 1,638 / 1,640 characters** (the length of the path varies), no `is_error` |
| Its content | `Error: result (1,000,000 characters across 1 line) exceeds maximum allowed tokens. Output has been saved to <session>/tool-results/mcp-fake_mcp-big-<ms>.txt.` then reading instructions (read the file in chunks, say what portion was read). **No preview** of the content itself |
| The whole result | written by the CLI to that file: 1,000,000 bytes |
| The turn | goes on: the model answers, `result` `success`, `terminal_reason` `completed`, `num_turns` 3, 0.0048 / 0.0057 USD |
| Before the call | the MCP tool is deferred: the model first loads it with `ToolSearch` (`select:mcp__fake_mcp__big`) |

So the megabyte never reaches the model nor the stream-json output (`tool_use_result` carries the
same notice), and the turn survives. The native harness does the same with another shape
(nexus GROS-1, #107): `<persisted-output>` with a 2 KB preview of the content and the path of the
file, ~2.2 KB in all.

## What would change the parity row

- the CLI inlining a megabyte, or cutting it in silence: `CLI_BIG_NOTICE` is no longer what the
  CLI does; replace it with the new record and re-run the scenario (a notice of 8 KB or more turns
  the cell red);
- the CLI failing the turn on such a result (`is_error`, `error_during_execution`): the
  transcript must replay that, and the cell becomes a declared gap.
