#!/bin/sh
# M2c, part A: a PreCompact COMMAND hook (settings file) on the real Claude Code CLI.
# usage: command-hook.sh <out-dir> reason|addctx|plain|none
# Needs an installed, logged-in `claude`. Seeds a session, then runs /compact with a
# PreCompact hook (precompact.sh) answering in the given mode. Positive when the compact
# summary (the isCompactSummary user message of compact-<mode>.jsonl) has ZEPHYR-42.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: $0 <out-dir> MODE}; MODE=${2:?usage: $0 <out-dir> MODE}
mkdir -p "$OUT/logs" "$OUT/proj"; OUT=$(cd "$OUT" && pwd)
export M2C_MODE=$MODE M2C_LOG=$OUT/logs
printf '{"hooks":{"PreCompact":[{"matcher":"","hooks":[{"type":"command","command":"%s"}]}]}}\n' \
  "$HERE/precompact.sh" > "$OUT/settings.json"
cd "$OUT/proj"
SID=$(uuidgen | tr A-Z a-z)
claude -p --model haiku --setting-sources project --settings "$OUT/settings.json" --session-id "$SID" \
  --output-format json "My favourite colour is teal and my dog is called Rufus. Reply with OK only." \
  < /dev/null > "$OUT/logs/seed-$MODE.json" 2>&1
claude -p --model haiku --setting-sources project --settings "$OUT/settings.json" --resume "$SID" \
  --output-format stream-json --verbose "/compact" < /dev/null > "$OUT/logs/compact-$MODE.jsonl" 2>&1
echo "mode=$MODE sid=$SID"
if grep -q ZEPHYR-42 "$OUT/logs/compact-$MODE.jsonl"; then echo "ZEPHYR-42 found (inspect the isCompactSummary message)"; else echo "ZEPHYR-42 absent"; fi
