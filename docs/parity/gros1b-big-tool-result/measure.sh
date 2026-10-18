#!/usr/bin/env bash
# GROS-1b: what the real Claude Code CLI does with an MCP tool result of about a megabyte.
#
# Needs an installed, logged-in `claude` and the `fake_mcp` of the nexus repository (the PO
# server of the parity scenario; its tool `big {n}` answers a text of n characters).
# Costs about 0.005 USD on claude-haiku-5-5. Run it again on each Claude Code CLI bump and
# compare with the README (`docs/parity/engines.md`, "À re-mesurer").
#
# Usage: ./measure.sh <fake_mcp> <out-dir>
set -euo pipefail

fake_mcp="${1:?usage: $0 <fake_mcp> <out-dir>}"
out="${2:?usage: $0 <fake_mcp> <out-dir>}"
mkdir -p "$out/work"
cat > "$out/mcp.json" <<EOF
{"mcpServers":{"fake_mcp":{"type":"stdio","command":"$fake_mcp","args":[]}}}
EOF

cd "$out/work"
claude --version > "$out/version.txt"
claude -p --model claude-haiku-5-5 \
  --strict-mcp-config --mcp-config "$out/mcp.json" \
  --allowedTools mcp__fake_mcp__big \
  --output-format stream-json --verbose \
  "Call the tool mcp__fake_mcp__big exactly once with n=1000000. Then, without calling any other tool, reply with: (1) the exact first 300 characters of the tool result you received, verbatim, (2) your estimate of the total length of what you received, (3) the word DONE." \
  < /dev/null > "$out/run.jsonl"

# What the model received for the call (its length and text), and how the turn ended.
jq -r 'select(.type=="user") | .message.content[0] | select(.content|type=="string")
       | "received \(.content|length) characters, is_error=\(.is_error)\n\(.content)"' "$out/run.jsonl"
jq -r 'select(.type=="result")
       | "result: subtype=\(.subtype) terminal_reason=\(.terminal_reason) num_turns=\(.num_turns) cost_usd=\(.total_cost_usd)"' "$out/run.jsonl"
