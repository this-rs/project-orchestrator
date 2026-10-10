#!/usr/bin/env bash
# Build `nexus-tools` (the files, shell and web tools of a native session) at the nexus revision
# Cargo.toml pins, with https (`--features tls`), into <root>/bin.
#
# Why here: a native session has no Read/Bash/WebFetch unless this executable is next to the
# server, on the PATH, or named by NEXUS_TOOLS_PATH (src/chat/config.rs, detect_nexus_tools_path).
# Every release channel ships it NEXT TO the `orchestrator` executable (archives, Homebrew, .deb,
# .rpm, Docker, desktop app); the CI builds it the same way (.github/actions/nexus-fakes), so the
# pinned revision is known to build with tls before a release needs it.
#
# The revision is the one of `nexus-claude` in Cargo.toml; every nexus dependency there must
# carry the same one (a server built against one revision ships the tools of that revision).
#
# Usage: scripts/build-nexus-tools.sh <root> [cargo install args...]   (e.g. --target <triple>)
set -euo pipefail

root="${1:?usage: $0 <root> [cargo install args...]}"
shift

manifest="${CARGO_MANIFEST:-Cargo.toml}"
revs="$(sed -n 's/^nexus-[a-z-]* = .*github.com\/this-rs\/nexus.*rev = "\([0-9a-f]\{40\}\)".*/\1/p' "$manifest" | sort -u)"
count="$(printf '%s\n' "$revs" | grep -c . || true)"
if [ "$count" -ne 1 ]; then
  echo "::error::expected exactly one nexus revision in $manifest, found $count: $revs" >&2
  exit 1
fi
rev="$revs"
echo "nexus-tools at this-rs/nexus@$rev (features: tls) -> $root/bin"

cargo install --git https://github.com/this-rs/nexus.git --rev "$rev" \
  --features tls --bin nexus-tools --root "$root" "$@" nexus-tools
