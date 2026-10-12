#!/usr/bin/env bash
# Write THIRD-PARTY-LICENSES.txt: the license texts and notices of the third-party code that the
# release binaries contain. MIT, BSD and Apache-2.0 all require them in a binary distribution, so
# every release channel (archives, .deb, .rpm, Docker, Homebrew, desktop app) ships this file next
# to LICENSE, LICENSE-MIT and LICENSE-EE.
#
# What goes in, one section each, in this order:
#   1. the Rust crates of this workspace's binaries (orchestrator, orch, mcp_server, desktop app):
#      cargo-about over the workspace, with about.toml (targets, accepted licenses, our own crates skipped);
#   2. nexus-tools, built from the this-rs/nexus revision Cargo.toml pins (scripts/build-nexus-tools.sh),
#      with the features it is built with: the LICENSE of that repository verbatim, then its crates;
#   3. every `--cargo` component given (another binary shipped next to the server);
#   4. the vendored notices of native libraries, licenses/third-party/<name>/ (ONNX Runtime...);
#   5. every `--notices` path given (e.g. the web interface's own dist/third-party-licenses.txt).
# For each Rust section, the NOTICE files of the crates (Apache-2.0 section 4(d)) are appended verbatim.
#
# Usage: scripts/third-party-licenses.sh [--offline] [--cargo NAME=MANIFEST[:FEATURES]]...
#                                        [--notices NAME=PATH]... <out-file>
#   --offline   no network: this workspace is read --frozen (Cargo.lock), the other components are
#               resolved from the local cargo cache. Run `cargo fetch` once before.
#   --cargo     a Rust binary built from another workspace: MANIFEST is the Cargo.toml of its package,
#               FEATURES (comma-separated) the features it is built with. The LICENSE*, NOTICE* and
#               COPYING* files at the root of its workspace are copied verbatim (Apache-2.0 4(d)).
#   --notices   a notice file, or a directory whose LICENSE*/NOTICE*/COPYING*/ThirdPartyNotices* files
#               are copied verbatim.
#
# Needs cargo, jq and cargo-about $ABOUT_VERSION (cargo install cargo-about --locked --features cli --version $ABOUT_VERSION).
set -euo pipefail

ABOUT_VERSION="0.9.2"

usage() { sed -n '/^# Usage:/,/^#   --notices/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

offline=0
cargo_components=()
notice_components=()
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --offline) offline=1 ;;
    --cargo) [ $# -ge 2 ] || usage; cargo_components+=("$2"); shift ;;
    --notices) [ $# -ge 2 ] || usage; notice_components+=("$2"); shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) [ -z "$out" ] || usage; out="$1" ;;
  esac
  shift
done
[ -n "$out" ] || usage

root="$(cd "$(dirname "$0")/.." && pwd)"
config="$root/about.toml"

got="$(cargo about --version 2>/dev/null || true)"
case "$got" in
  *" $ABOUT_VERSION") ;;
  *) echo "::error::cargo-about $ABOUT_VERSION is required (found: ${got:-none}): cargo install cargo-about --locked --features cli --version $ABOUT_VERSION" >&2; exit 1 ;;
esac
command -v jq >/dev/null || { echo "::error::jq is required" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
body="$tmp/body.txt"
: > "$body"
section=0
titles=()

rule() { printf '%s\n' "================================================================================"; }

begin_section() {
  section=$((section + 1))
  titles+=("$section. $1")
  { echo; rule; printf '%d. %s\n' "$section" "$1"; rule; echo; } >> "$body"
}

# Copy a file verbatim, under a header naming it.
verbatim() { # <label> <file>
  { printf -- '--- %s ---\n\n' "$1"; cat "$2"; printf '\n\n'; } >> "$body"
}

# The license/notice files at the top of a directory, sorted.
notice_files() { # <dir>
  find "$1" -maxdepth 1 -type f \( -iname 'LICENSE*' -o -iname 'LICENCE*' -o -iname 'NOTICE*' \
    -o -iname 'COPYING*' -o -iname 'ThirdPartyNotices*' \) | LC_ALL=C sort
}

# Render cargo-about's JSON as text, then append the crates' NOTICE files.
render_crates() { # <json>
  jq -r '
    "Licenses, by number of crates:",
    (.overview[] | "  \(.count)\t\(.name) (\(.id))"),
    "",
    (.licenses[] |
      "--------------------------------------------------------------------------------",
      "\(.name) (\(.id))",
      "Used by:",
      (.used_by[] | "  - \(.crate.name) \(.crate.version)" + (if .crate.repository then " <\(.crate.repository)>" else "" end)),
      "",
      .text,
      "")
  ' "$1" >> "$body"
  local m d f any=0
  while IFS= read -r m; do
    d="$(dirname "$m")"
    while IFS= read -r f; do
      case "$(basename "$f")" in [Nn][Oo][Tt][Ii][Cc][Ee]*) ;; *) continue ;; esac
      if [ "$any" -eq 0 ]; then
        printf -- '--------------------------------------------------------------------------------\nNOTICE files of the crates above (Apache-2.0, section 4(d)), verbatim\n\n' >> "$body"
        any=1
      fi
      verbatim "$(jq -r --arg m "$m" '.crates[] | select(.package.manifest_path == $m) | "\(.package.name) \(.package.version)"' "$1" | head -1): $(basename "$f")" "$f"
    done < <(notice_files "$d")
  done < <(jq -r '.crates[].package.manifest_path' "$1" | LC_ALL=C sort -u)
}

# cargo-about over one package (or this workspace), JSON out.
about_json() { # <json-out> <cargo-about args...>
  local json="$1"; shift
  if ! cargo about generate --format json -c "$config" --fail "$@" -o "$json" 2> "$tmp/about.err"; then
    cat "$tmp/about.err" >&2
    echo "::error::cargo-about failed ($*)" >&2
    exit 1
  fi
}

# The root of the cargo workspace a manifest belongs to.
workspace_root() { # <manifest>
  local d; d="$(cd "$(dirname "$1")" && pwd)"
  while [ "$d" != "/" ]; do
    if [ -f "$d/Cargo.toml" ] && grep -q '^\[workspace\]' "$d/Cargo.toml"; then echo "$d"; return; fi
    d="$(dirname "$d")"
  done
  dirname "$1"
}

# A Rust binary from another workspace: its workspace's notices verbatim, then its crates.
cargo_component() { # <name> <manifest> <features>
  local name="$1" manifest="$2" features="$3" ws copy rel f n
  [ -f "$manifest" ] || { echo "::error::$name: no manifest at $manifest" >&2; exit 1; }
  ws="$(workspace_root "$manifest")"
  begin_section "$name"
  while IFS= read -r f; do verbatim "$(basename "$f") (root of its workspace)" "$f"; done < <(notice_files "$ws")
  # Resolve in a copy: a cargo git checkout is not ours to write a Cargo.lock into.
  n=$((section))
  copy="$tmp/ws$n"
  mkdir -p "$copy"
  (cd "$ws" && tar --exclude ./target -cf - .) | (cd "$copy" && tar -xf -)
  rel="${manifest#"$ws"/}"
  local args=(-m "$copy/$rel")
  [ -n "$features" ] && args+=(--features "${features//,/ }")
  [ "$offline" -eq 1 ] && args+=(--offline)
  about_json "$tmp/c$n.json" "${args[@]}"
  render_crates "$tmp/c$n.json"
}

# --- 1. this workspace ------------------------------------------------------------------------------
begin_section "Rust crates of Project Orchestrator (orchestrator, orch, mcp_server, desktop app)"
ws_args=(--workspace -m "$root/Cargo.toml")
[ "$offline" -eq 1 ] && ws_args+=(--frozen) || ws_args+=(--locked)
about_json "$tmp/ws.json" "${ws_args[@]}"
render_crates "$tmp/ws.json"

# --- 2. nexus-tools, at the pinned nexus revision, with tls (as scripts/build-nexus-tools.sh builds it)
meta_args=(--format-version 1 --manifest-path "$root/Cargo.toml")
[ "$offline" -eq 1 ] && meta_args+=(--frozen) || meta_args+=(--locked)
nexus_tools="$(cargo metadata "${meta_args[@]}" 2>/dev/null \
  | jq -r '.packages[] | select(.name == "nexus-tools" and (.source // "" | startswith("git+"))) | .manifest_path' | head -1)"
[ -n "$nexus_tools" ] || { echo "::error::nexus-tools not found in the dependency graph of $root/Cargo.toml" >&2; exit 1; }
cargo_component "nexus-tools (this-rs/nexus, features: tls)" "$nexus_tools" "tls"

# --- 3. other Rust binaries shipped next to the server -----------------------------------------------
for c in ${cargo_components[@]+"${cargo_components[@]}"}; do
  name="${c%%=*}"; spec="${c#*=}"
  [ "$name" != "$c" ] || { echo "::error::--cargo expects NAME=MANIFEST[:FEATURES], got: $c" >&2; exit 2; }
  manifest="${spec%%:*}"; features=""
  [ "$manifest" = "$spec" ] || features="${spec#*:}"
  cargo_component "$name" "$manifest" "$features"
done

# --- 4/5. native libraries (vendored notices) and the given notice paths -----------------------------
notice_component() { # <name> <path>
  local name="$1" path="$2" f
  begin_section "$name"
  if [ -d "$path" ]; then
    local found=0
    while IFS= read -r f; do verbatim "$(basename "$f")" "$f"; found=1; done < <(notice_files "$path")
    [ "$found" -eq 1 ] || { echo "::error::$name: no license or notice file in $path" >&2; exit 1; }
  elif [ -f "$path" ]; then
    verbatim "$(basename "$path")" "$path"
  else
    echo "::error::$name: $path does not exist" >&2; exit 1
  fi
}

for d in "$root"/licenses/third-party/*/; do
  [ -d "$d" ] || continue
  label="$(basename "$d")"
  [ -f "$d/NAME" ] && label="$(head -1 "$d/NAME")"
  notice_component "$label" "$d"
done

for c in ${notice_components[@]+"${notice_components[@]}"}; do
  name="${c%%=*}"; path="${c#*=}"
  [ "$name" != "$c" ] || { echo "::error::--notices expects NAME=PATH, got: $c" >&2; exit 2; }
  notice_component "$name" "$path"
done

# --- assemble -----------------------------------------------------------------------------------------
mkdir -p "$(dirname "$out")"
{
  echo "THIRD-PARTY SOFTWARE NOTICES — Project Orchestrator"
  echo
  echo "The Project Orchestrator binaries contain third-party software, listed below with the"
  echo "license texts and notices their licenses require. Project Orchestrator itself is licensed"
  echo "under LICENSE (MIT AND BUSL-1.1, see LICENSE-MIT and LICENSE-EE), shipped alongside."
  echo
  echo "Generated by scripts/third-party-licenses.sh (cargo-about $ABOUT_VERSION)."
  echo
  echo "Sections:"
  printf "  %s\n" "${titles[@]}"
  cat "$body"
} > "$out"
echo "wrote $out ($section sections, $(wc -c < "$out" | tr -d ' ') bytes)"
