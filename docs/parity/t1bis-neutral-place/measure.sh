#!/usr/bin/env bash
# T1-bis: does the real Claude Code CLI resume a session whose cwd is a NEUTRAL directory of the
# host (`src/chat/neutral_place.rs`), after that directory was removed and made again?
#
# What the host does, replayed by hand: a session opened without `cwd` runs in
# `<config dir>/project-orchestrator/chat-neutral/<session id>` (empty, 0700, no `.claude/`); the
# directory is removed when the session closes (or swept after 24 h) and recreated, at the SAME
# path, when the session is resumed (`neutral_place::ensure`). Claude Code keys its transcripts on
# the cwd, so the resume only works if the path is stable.
#
# Needs an installed, logged-in `claude`. Two turns on claude-haiku-5-5, no MCP server, no tool:
# about 0.01 USD. Run it again on each Claude Code CLI bump.
#
# Usage: ./measure.sh <out-dir> [<neutral root>]
#   <neutral root> defaults to the host's real root (macOS: ~/Library/Application Support/
#   project-orchestrator/chat-neutral; Linux: ${XDG_CONFIG_HOME:-~/.config}/project-orchestrator/
#   chat-neutral).
set -euo pipefail

out="${1:?usage: $0 <out-dir> [<neutral root>]}"
if [[ $# -ge 2 ]]; then
  root="$2"
elif [[ "$(uname)" == Darwin ]]; then
  root="$HOME/Library/Application Support/project-orchestrator/chat-neutral"
else
  root="${XDG_CONFIG_HOME:-$HOME/.config}/project-orchestrator/chat-neutral"
fi
mkdir -p "$out"
sid_host="$(uuidgen | tr '[:upper:]' '[:lower:]')"
dir="$root/$sid_host"
word="PAPAYE-$RANDOM$RANDOM"
model="claude-haiku-5-5"
common=(--model "$model" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --output-format json)

claude --version > "$out/version.txt"

# What Claude Code could read above the neutral directory: a CLAUDE.md or a .claude/ in an
# ancestor (the user's own ~/.claude is read by every session, wherever it runs).
{
  p="$root"
  while [[ "$p" != "/" ]]; do
    for f in "$p/CLAUDE.md" "$p/CLAUDE.local.md" "$p/.claude"; do
      [[ -e "$f" ]] && echo "found: $f"
    done
    p="$(dirname "$p")"
  done
  echo "ancestors checked up to /"
} > "$out/ancestors.txt"

# Turn 1, in a freshly made neutral directory (what `ensure` makes).
mkdir -p "$dir" && chmod 700 "$dir"
(cd "$dir" && claude -p "${common[@]}" \
  "Remember this word for later: $word. Reply with OK only." < /dev/null) > "$out/turn1.json"
cli_sid="$(jq -r .session_id "$out/turn1.json")"
echo "host session $sid_host, cli session $cli_sid, word $word" > "$out/ids.txt"
ls -A "$dir" > "$out/dir-after-turn1.txt"

# The session closes: the directory goes (`remove_for`). Then a resume makes it again, empty.
rm -rf "$dir"
mkdir -p "$dir" && chmod 700 "$dir"

# Turn 2: --resume from the same (recreated) path.
(cd "$dir" && claude -p "${common[@]}" --resume "$cli_sid" \
  "What word did I ask you to remember? Reply with the word only." < /dev/null) > "$out/turn2.json"
rm -rf "$dir"

answer="$(jq -r .result "$out/turn2.json")"
jq -r '"turn 1: subtype=\(.subtype) is_error=\(.is_error) cost_usd=\(.total_cost_usd)"' "$out/turn1.json"
jq -r '"turn 2: subtype=\(.subtype) is_error=\(.is_error) session=\(.session_id) cost_usd=\(.total_cost_usd)"' "$out/turn2.json"
echo "expected $word, answered: $answer"
cat "$out/ancestors.txt"
if [[ "$answer" == *"$word"* ]]; then echo "RESUME OK"; else echo "RESUME LOST THE CONVERSATION"; exit 1; fi
