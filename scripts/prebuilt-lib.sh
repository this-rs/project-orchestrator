#!/usr/bin/env bash
# Versioned prebuilt native libraries: build once, download everywhere.
#
#   scripts/prebuilt-lib.sh use     <lib> <target>          # CI step: fetch + export the build env
#   scripts/prebuilt-lib.sh fetch   <lib> <target> <dest>   # materialise one library, --no-build to forbid building
#   scripts/prebuilt-lib.sh publish <lib> <target> <dir>    # upload it to the library release
#   scripts/prebuilt-lib.sh build   <lib> <target> <dir>    # build it here (the prebuilt-libs workflow)
#   scripts/prebuilt-lib.sh mirror  <lib> <target>          # fetch from upstream + publish
#   scripts/prebuilt-lib.sh plan-build|plan-mirror [--json] [--force]
#   scripts/prebuilt-lib.sh version <lib>
#
# THREE SOURCES, in this order — the release never depends on a warm cache or on a network hiccup:
#
#   1. the library release `lib/<lib>-<version>` of this repository. A release asset lives as long
#      as its tag does, is versioned, and is visible to a TAG-TRIGGERED run — which actions/cache is
#      NOT (a tag build starts with an empty cache) and which GitHub evicts after 7 idle days.
#   2. the upstream archive (Microsoft, for ONNX Runtime).
#   3. a local build, only for the (lib, target) pairs upstream stopped publishing — macOS Intel for
#      ort from 1.24 on — and only where the host IS the target (a cross-built V8-style snapshot or
#      ONNX Runtime is the wrong architecture at runtime).
#
# Publishing is idempotent: an existing asset is left alone unless --force. The library release is
# marked prerelease so it can never become "latest" (install.sh and Homebrew read /releases/latest).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
# shellcheck source=native-libs.env
. "$HERE/native-libs.env"

log() { printf 'prebuilt-lib: %s\n' "$*"; }

# Scratch dirs are cleaned once, at exit: a RETURN trap would fire on every nested function
# call (log, tar...) and delete the directory the caller is still using.
TMPDIRS=()
mk_tmp() { local d; d=$(mktemp -d); TMPDIRS+=("$d"); printf "%s\n" "$d"; }
cleanup() { local d; for d in ${TMPDIRS[@]+"${TMPDIRS[@]}"}; do [ -n "$d" ] && rm -rf "$d"; done; }
trap cleanup EXIT
warn() { printf '::warning::prebuilt-lib: %s\n' "$*" >&2; }
die() { printf '::error::prebuilt-lib: %s\n' "$*" >&2; exit 1; }

lib_version() {
  case "$1" in
    ort) printf '%s\n' "$ORT_VERSION" ;;
    *) die "unknown library '$1' (known: ort)" ;;
  esac
}

lib_tag() { printf 'lib/%s-%s\n' "$1" "$2"; }
lib_asset() { printf '%s-%s-%s.tar.gz\n' "$1" "$2" "$3"; }

repo_slug() {
  if [ -n "${GITHUB_REPOSITORY:-}" ]; then printf '%s\n' "$GITHUB_REPOSITORY"; return 0; fi
  command -v gh >/dev/null || return 1
  ( cd "$ROOT" && gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null )
}

# The archive upstream publishes for (target), or empty when upstream has none.
upstream_url() {
  local lib=$1 version=$2 target=$3
  case "$lib:${target}" in
    ort:x86_64-unknown-linux-gnu)   printf 'https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-linux-x64-%s.tgz\n' "$version" "$version" ;;
    ort:aarch64-unknown-linux-gnu)  printf 'https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-linux-aarch64-%s.tgz\n' "$version" "$version" ;;
    ort:aarch64-apple-darwin)       printf 'https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-osx-arm64-%s.tgz\n' "$version" "$version" ;;
    *) : ;;
  esac
}

# Can this script build (lib, target) here? Only where host == target and a recipe exists.
can_build() {
  local lib=$1 target=$2
  case "$lib:$target" in
    ort:x86_64-apple-darwin)
      [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = x86_64 ] ;;
    *) return 1 ;;
  esac
}

runs_on() {
  case "$1" in
    x86_64-apple-darwin) printf 'macos-15-intel\n' ;;
    *) printf 'ubuntu-22.04\n' ;;
  esac
}

# ---------------------------------------------------------------- sources

from_library_release() {
  local lib=$1 version=$2 target=$3 dest=$4 repo tag asset tmp
  command -v gh >/dev/null 2>&1 || { log "$lib $version $target: no gh, library release skipped"; return 1; }
  repo=$(repo_slug) || { log "$lib $version $target: no repository, library release skipped"; return 1; }
  tag=$(lib_tag "$lib" "$version"); asset=$(lib_asset "$lib" "$version" "$target")
  tmp=$(mk_tmp)
  if GH_TOKEN="${GH_TOKEN:-}" gh release download "$tag" -R "$repo" -p "$asset" -D "$tmp" >/dev/null 2>&1 \
     && [ -s "$tmp/$asset" ]; then
    rm -rf "$dest" && mkdir -p "$dest"
    tar xzf "$tmp/$asset" -C "$dest"
    log "$lib $version $target <- $repo@$tag (prebuilt asset, no build)"
    return 0
  fi
  log "$lib $version $target: not published on $tag yet"
  return 1
}

from_upstream() {
  local lib=$1 version=$2 target=$3 dest=$4 url tmp
  url=$(upstream_url "$lib" "$version" "$target")
  [ -n "$url" ] || { log "$lib $version $target: upstream publishes nothing"; return 1; }
  tmp=$(mk_tmp)
  if curl -fsSL --retry 3 --retry-delay 2 -o "$tmp/a.tgz" "$url"; then
    rm -rf "$dest" && mkdir -p "$dest"
    tar xzf "$tmp/a.tgz" -C "$dest" --strip-components=1
    log "$lib $version $target <- upstream ($url)"
    return 0
  fi
  warn "$lib $version $target: upstream download failed ($url)"
  return 1
}

build_local() {
  local lib=$1 version=$2 target=$3 dest=$4
  can_build "$lib" "$target" || die "$lib $version $target: no build recipe on $(uname -s)/$(uname -m) — publish it from the prebuilt-libs workflow first"
  case "$lib:$target" in
    ort:x86_64-apple-darwin) "$HERE/build-onnxruntime-macos.sh" "$version" "$dest" ;;
    *) die "no build recipe for $lib/$target" ;;
  esac
  log "$lib $version $target: built here"
}

fetch_into() {
  local lib=$1 target=$2 dest=$3 allow_build=$4 version
  version=$(lib_version "$lib")
  mkdir -p "$(dirname "$dest")"
  from_library_release "$lib" "$version" "$target" "$dest" && return 0
  from_upstream "$lib" "$version" "$target" "$dest" && return 0
  [ "$allow_build" = 1 ] || die "$lib $version $target: no prebuilt and building is forbidden here"
  build_local "$lib" "$version" "$target" "$dest"
}

# ---------------------------------------------------------------- publish

ensure_release() {
  local lib=$1 version=$2 tag repo
  tag=$(lib_tag "$lib" "$version"); repo=$(repo_slug) \
    || die "cannot tell which repository to publish to (set GITHUB_REPOSITORY)"
  if gh release view "$tag" -R "$repo" >/dev/null 2>&1; then return 0; fi
  log "creating library release $tag"
  # --prerelease: a library release must never become `/releases/latest` (install.sh and the
  # Homebrew formula read it). The tag namespace `lib/` keeps it out of `v*` (release.yml trigger).
  gh release create "$tag" -R "$repo" --prerelease \
    --title "Native library: $lib $version" \
    --notes "Prebuilt artifacts consumed by .github/workflows/release.yml (scripts/prebuilt-lib.sh).
This is NOT a product release: it is not tagged \`v*\`, it is marked prerelease, and it carries no changelog." \
    >/dev/null 2>&1 || true
  gh release view "$tag" -R "$repo" >/dev/null 2>&1 || die "library release $tag could not be created"
}

publish_dir() {
  local lib=$1 target=$2 dir=$3 force=$4 version tag asset repo tmp
  version=$(lib_version "$lib"); repo=$(repo_slug) || die "no repository to publish to"
  tag=$(lib_tag "$lib" "$version"); asset=$(lib_asset "$lib" "$version" "$target")
  ensure_release "$lib" "$version"
  if [ "$force" != 1 ] && gh release view "$tag" -R "$repo" --json assets \
       --jq '.assets[].name' 2>/dev/null | grep -qxF "$asset"; then
    log "$lib $version $target: already published on $tag, left alone"
    return 0
  fi
  tmp=$(mk_tmp)
  tar czf "$tmp/$asset" -C "$dir" .
  ( cd "$tmp" && sha256sum "$asset" > "$asset.sha256" 2>/dev/null || shasum -a 256 "$asset" > "$asset.sha256" )
  gh release upload "$tag" -R "$repo" "$tmp/$asset" "$tmp/$asset.sha256" --clobber >/dev/null \
    || die "upload of $asset to $tag failed"
  log "$lib $version $target: published $asset to $tag ($(du -h "$tmp/$asset" | cut -f1))"
}

# ---------------------------------------------------------------- plans

# Every (lib, target) the release consumes, from the pins.
ALL_PAIRS='ort:x86_64-unknown-linux-gnu ort:aarch64-unknown-linux-gnu ort:aarch64-apple-darwin ort:x86_64-apple-darwin'

is_published() {
  local lib=$1 version=$2 target=$3 repo tag asset
  repo=$(repo_slug) || return 1
  tag=$(lib_tag "$lib" "$version"); asset=$(lib_asset "$lib" "$version" "$target")
  gh release view "$tag" -R "$repo" --json assets --jq '.assets[].name' 2>/dev/null | grep -qxF "$asset"
}

plan() {
  local what=$1 json=$2 force=$3
  local out='' sep='' lib target version url runner
  for pair in $ALL_PAIRS; do
    lib=${pair%%:*}; target=${pair#*:}
    version=$(lib_version "$lib")
    url=$(upstream_url "$lib" "$version" "$target")
    if [ "$what" = build ]; then
      # Only the pairs upstream cannot serve, and only when we cannot build them on any runner
      # we know — a build job is a macos-15-intel (Intel) or a linux runner; keep it explicit.
      [ -z "$url" ] || continue
    else
      [ -n "$url" ] || continue
    fi
    if [ "$force" != 1 ] && is_published "$lib" "$version" "$target"; then continue; fi
    runner=$(runs_on "$target")
    if [ "$json" = 1 ]; then
      out="$out$sep{\"lib\":\"$lib\",\"version\":\"$version\",\"target\":\"$target\",\"runner\":\"$runner\"}"
      sep=','
    else
      printf '%s %s %s (%s)\n' "$lib" "$version" "$target" "$runner"
    fi
  done
  [ "$json" = 1 ] && printf '[%s]\n' "$out"
  return 0
}

# ---------------------------------------------------------------- commands

cmd=${1:-}; shift || true
case "$cmd" in
  version)   lib_version "$1" ;;
  targets)   for p in $ALL_PAIRS; do [ "${p%%:*}" = "$1" ] && printf '%s\n' "${p#*:}"; done ;;
  fetch)
    lib=$1; target=$2; dest=$3; shift 3; allow=1
    for a in "$@"; do [ "$a" = --no-build ] && allow=0; done
    fetch_into "$lib" "$target" "$dest" "$allow" ;;
  build)
    lib=$1; target=$2; dir=$3
    rm -rf "$dir"; mkdir -p "$dir"
    build_local "$lib" "$(lib_version "$lib")" "$target" "$dir" ;;
  publish)
    lib=$1; target=$2; dir=$3; shift 3; force=0
    for a in "$@"; do [ "$a" = --force ] && force=1; done
    publish_dir "$lib" "$target" "$dir" "$force" ;;
  mirror)
    lib=$1; target=$2; tmp=$(mk_tmp)
    from_upstream "$lib" "$(lib_version "$lib")" "$target" "$tmp" \
      || die "$lib $target: upstream has no archive, nothing to mirror"
    publish_dir "$lib" "$target" "$tmp" 1 ;;
  plan-build)  json=0; force=0; for a in "$@"; do case "$a" in --json) json=1 ;; --force) force=1 ;; esac; done; plan build "$json" "$force" ;;
  plan-mirror) json=0; force=0; for a in "$@"; do case "$a" in --json) json=1 ;; --force) force=1 ;; esac; done; plan mirror "$json" "$force" ;;
  use)
    lib=$1; target=$2
    version=$(lib_version "$lib")
    : "${RUNNER_TEMP:=${TMPDIR:-/tmp}}"
    dest="$RUNNER_TEMP/prebuilt/$lib-$version-$target"
    fetch_into "$lib" "$target" "$dest" 1
    case "$lib" in
      ort)
        "$HERE/check-ort-api.sh" "$dest"
        printf 'ORT_LIB_LOCATION=%s/lib\n' "$dest" >> "${GITHUB_ENV:-/dev/null}"
        printf 'ORT_PREFER_DYNAMIC_LINK=1\n'     >> "${GITHUB_ENV:-/dev/null}"
        # cross-rs mounts only the workspace by default: hand it the library too.
        [ "$target" = aarch64-unknown-linux-gnu ] \
          && printf 'CROSS_CONTAINER_OPTS=-v %s:%s\n' "$dest" "$dest" >> "${GITHUB_ENV:-/dev/null}"
        log "ort $version $target ready in $dest/lib"
        ls -la "$dest/lib" ;;
    esac ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2 ;;
esac
