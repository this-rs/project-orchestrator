#!/usr/bin/env bash
# Build ONNX Runtime as a shared library for macOS Intel (x86_64-apple-darwin).
#
#   scripts/build-onnxruntime-macos.sh <version> <out dir>
#
# Why we build it: fastembed turns on ort's `api-24`, and ort panics at start on a runtime that
# does not serve that API (scripts/check-ort-api.sh). Microsoft publishes no osx-x86_64 build
# from 1.24 on, and ort-sys (pyke.io) has no prebuilt for this target. So the release builds it
# here, on the native Intel runner (host == target), and caches it per version (release.yml).
#
# Output: the layout of Microsoft's archives — <out>/lib/libonnxruntime.<v>.dylib (+ the
# libonnxruntime.dylib link), <out>/include/onnxruntime_*.h, LICENSE, ThirdPartyNotices.txt.
#
# The macOS floor is MACOSX_DEPLOYMENT_TARGET (default 13.4, the promise of check-os-floor.sh
# for macos-x86_64). Microsoft builds 1.24 for 14.0; if 1.24 stops compiling for 13.4, raising
# this means raising the promise too (and the Macs below go to the legacy build).
set -euo pipefail

version="${1:?usage: build-onnxruntime-macos.sh <version> <out dir>}"
out="${2:?usage: build-onnxruntime-macos.sh <version> <out dir>}"
floor="${MACOSX_DEPLOYMENT_TARGET:-13.4}"
export MACOSX_DEPLOYMENT_TARGET="$floor"

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = x86_64 ] \
  || { echo "build-onnxruntime-macos: run on an Intel Mac (host == target)" >&2; exit 2; }

src="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/onnxruntime-src-${version}"
rm -rf "$src"
git clone --quiet --depth 1 --branch "v${version}" --recurse-submodules --shallow-submodules \
  https://github.com/microsoft/onnxruntime "$src"

echo "build-onnxruntime-macos: ONNX Runtime ${version}, x86_64, macOS ${floor}, $(sysctl -n hw.ncpu) cores"
( cd "$src" && ./build.sh \
    --config Release \
    --build_dir "$src/b" \
    --build_shared_lib \
    --parallel \
    --skip_tests \
    --skip_submodule_sync \
    --compile_no_warning_as_error \
    --cmake_extra_defines \
      CMAKE_OSX_ARCHITECTURES=x86_64 \
      CMAKE_OSX_DEPLOYMENT_TARGET="$floor" \
      onnxruntime_BUILD_UNIT_TESTS=OFF )

rel="$src/b/Release"
lib="$rel/libonnxruntime.${version}.dylib"
[ -f "$lib" ] || { echo "build-onnxruntime-macos: no $lib" >&2; ls -la "$rel" >&2; exit 1; }

rm -rf "$out" && mkdir -p "$out/lib" "$out/include"
cp "$lib" "$out/lib/"
ln -s "libonnxruntime.${version}.dylib" "$out/lib/libonnxruntime.dylib"
cp "$src"/include/onnxruntime/core/session/onnxruntime_*.h "$out/include/"
cp "$src/LICENSE" "$src/ThirdPartyNotices.txt" "$src/VERSION_NUMBER" "$out/"
strip -x "$out/lib/libonnxruntime.${version}.dylib"

echo "build-onnxruntime-macos: done"
ls -la "$out/lib"
otool -D "$out/lib/libonnxruntime.${version}.dylib"
otool -l "$out/lib/libonnxruntime.${version}.dylib" | grep -A4 LC_BUILD_VERSION | grep -E 'minos|sdk'
