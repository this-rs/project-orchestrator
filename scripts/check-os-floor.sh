#!/usr/bin/env bash
# Refuse an artifact whose REAL minimum system is higher than the one we promise.
#
# Why: 0.0.17 shipped a macOS Intel app that declared "macOS 10.15" while one dylib inside (ONNX Runtime) needed 13.4, so it
# crashed at launch on Catalina. The promise and the files were never compared. This compares them, on every Mach-O / ELF
# found under the given paths, and fails the build when a file asks for more than the promise.
#
# Usage: scripts/check-os-floor.sh <id> <file-or-dir>...
#   id: macos-arm64 | macos-x86_64 | macos-x86_64-legacy | linux-x86_64 | linux-arm64 | linux-desktop-x86_64 | linux-desktop-arm64
#
# The promises live in PROMISE below, nowhere else. Changing one is a product decision (it changes who can install the
# app): update the README and the site (content/downloads.ts) in the same change.
set -euo pipefail

id="${1:?usage: $0 <id> <file-or-dir>...}"
shift

# macOS: minimum macOS version.  Linux: highest glibc symbol version (GLIBC_x.y) the binaries may need.
case "$id" in
  macos-arm64)         kind=macos;  promise=11.0  ;;
  macos-x86_64)        kind=macos;  promise=13.4  ;; # ONNX Runtime (libonnxruntime) is built for 13.4
  macos-x86_64-legacy) kind=macos;  promise=10.15 ;; # no ONNX Runtime at all
  # Linux, measured with objdump -T on the real files and confirmed by a dry run of the release workflow:
  #  - x86_64 and the desktop apps embed a STATIC ONNX Runtime (pyke) built against glibc 2.38+: they need 2.39 and cannot even link
  #    on an Ubuntu 22.04 runner. Ubuntu 24.04+, Debian 13+, Fedora 40+.
  #  - the arm64 CLI archive uses the dynamic Microsoft ONNX Runtime (glibc 2.27) and is built natively on Ubuntu 22.04: 2.35
  #    (Ubuntu 22.04+, Debian 12+, Fedora 36+).
  # Lowering the 2.39 ones = dynamic ONNX Runtime + shipping it in the .deb/.rpm + packaging dependencies: a separate piece of work.
  linux-x86_64)         kind=linux;  promise=2.39  ;;
  linux-arm64)          kind=linux;  promise=2.35  ;;
  linux-desktop-x86_64) kind=linux;  promise=2.39  ;;
  linux-desktop-arm64)  kind=linux;  promise=2.39  ;;
  *) echo "::error::unknown id '$id'" >&2; exit 2 ;;
esac

# highest($a $b) -> the greater version (version sort).
highest() { printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1; }

# Every regular file under the given paths.
files=()
while IFS= read -r f; do files+=("$f"); done < <(find "$@" -type f 2>/dev/null)
[ "${#files[@]}" -gt 0 ] || { echo "::error::no file under: $*" >&2; exit 2; }

checked=0
failed=0
worst="0"

for f in "${files[@]}"; do
  desc="$(file -b "$f" 2>/dev/null || true)"
  case "$kind" in
    macos)
      case "$desc" in *Mach-O*) ;; *) continue ;; esac
      # Rust binaries carry LC_VERSION_MIN_MACOSX ("version"), newer toolchains LC_BUILD_VERSION ("minos"): read both.
      floor="$(vtool -show-build "$f" 2>/dev/null | awk '$1=="minos" || $1=="version" {print $2; exit}')"
      if [ -z "$floor" ]; then
        echo "::error::$f: no minimum macOS version found in the Mach-O headers" >&2; failed=1; continue
      fi
      ;;
    linux)
      case "$desc" in *ELF*) ;; *) continue ;; esac
      floor="$(grep -a -o 'GLIBC_[0-9][0-9.]*' "$f" | sed 's/GLIBC_//' | sort -uV | tail -1 || true)"
      [ -n "$floor" ] || floor="0" # static or no versioned glibc symbol: no constraint
      ;;
  esac
  checked=$((checked + 1))
  worst="$(highest "$worst" "$floor")"
  if [ "$(highest "$floor" "$promise")" != "$promise" ]; then
    echo "::error::$f needs $kind $floor, above the promised $promise ($id)" >&2
    failed=1
  else
    echo "ok   $floor  $f"
  fi
done

# The legacy macOS build exists to run WITHOUT ONNX Runtime: its presence anywhere means the build is not the legacy one.
if [ "$id" = "macos-x86_64-legacy" ]; then
  while IFS= read -r f; do
    echo "::error::ONNX Runtime found in the legacy build: $f" >&2; failed=1
  done < <(find "$@" -iname 'libonnxruntime*' 2>/dev/null)
  for f in "${files[@]}"; do
    case "$(file -b "$f" 2>/dev/null)" in *Mach-O*) otool -L "$f" 2>/dev/null | grep -qi onnxruntime && { echo "::error::$f links ONNX Runtime" >&2; failed=1; } ;; esac
  done
fi

# What the .app TELLS macOS must not be lower than what its files need. This is the exact 0.0.17 lie: LSMinimumSystemVersion 10.15
# on an app holding a 13.4 dylib. (Apple Silicon cannot run below 11.0 whatever the plist says, so it is not checked there.)
if [ "$kind" = "macos" ] && [ "$id" != "macos-arm64" ]; then
  while IFS= read -r plist; do
    declared="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' "$plist" 2>/dev/null || true)"
    if [ -z "$declared" ]; then
      echo "::error::$plist declares no LSMinimumSystemVersion" >&2; failed=1
    elif [ "$(highest "$declared" "$worst")" != "$declared" ]; then
      echo "::error::$plist declares macOS $declared but its files need $worst: the app would crash at launch instead of being refused" >&2
      failed=1
    else
      echo "ok   declared $declared >= needed $worst  $plist"
    fi
  done < <(find "$@" -path '*.app/Contents/Info.plist' 2>/dev/null)
fi

[ "$checked" -gt 0 ] || { echo "::error::no $kind executable found under: $*" >&2; exit 2; }
echo "$id: $checked file(s), highest requirement $worst, promise $promise"
exit "$failed"
