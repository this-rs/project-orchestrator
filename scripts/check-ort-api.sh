#!/usr/bin/env bash
# Does an ONNX Runtime we link dynamically serve the API version `ort` was compiled for?
#
#   scripts/check-ort-api.sh <onnxruntime dir>   (the extracted archive: include/ and lib/)
#
# `ort` asks the runtime for ONE API version at start-up (ort_sys::ORT_API_VERSION, set by the
# highest `api-N` feature of ort-sys, which fastembed turns on) and PANICS when the runtime does
# not serve it ("Failed to initialize ORT API"). A runtime that is too old links fine and only
# fails when local embeddings start. Measured: ort 2.0.0-rc.12 + fastembed 5.17.2 ask for API 24;
# ONNX Runtime 1.23.2 serves 1..23 (`GetApi(24)` returns NULL) — what the Linux ARM64 and macOS
# Intel builds shipped from 0.0.15 to 0.0.17.
#
# TWO FACTS, TWO SOURCES:
#   - what the runtime SERVES: ORT_API_VERSION in its own header — the value it was built with,
#     i.e. the same number GetApi() answers. No compiler, no cargo.
#   - what `ort` ASKS FOR: ORT_REQUIRED_API in scripts/native-libs.env, mirroring fastembed's
#     `api-N` feature. The script must not read it from cargo: the release workflow rewrites
#     `version` in Cargo.toml before building, which makes Cargo.lock stale by design, and
#     `cargo metadata --locked` then fails ("cannot update the lock file ... because --locked was
#     passed"). A version of this check that needed cargo killed the Linux build 19 seconds in
#     (dry run 39135216528, 18/10/2026) — the pin is the source of truth.
#
# The cargo lookup is kept as a ONE-WAY courtesy: when it works (a pristine tree, a local run) it
# can only ADD a failure, by noticing that the resolved features disagree with the pin.
set -euo pipefail

dir="${1:?usage: check-ort-api.sh <onnxruntime dir>}"
header="$dir/include/onnxruntime_c_api.h"
[ -f "$header" ] || { echo "check-ort-api: no $header" >&2; exit 2; }

here=$(cd "$(dirname "$0")" && pwd)
# scripts/native-libs.env is the single source of the pin (scripts/prebuilt-lib.sh reads the same
# file). The literal below is only a fallback for a tree where the file is absent: 24 is what
# fastembed 5.17.2 asks for.
if [ -f "$here/native-libs.env" ]; then
  # shellcheck source=native-libs.env
  . "$here/native-libs.env"
fi
required="${ORT_REQUIRED_API:-24}"

served="$(sed -n 's/^#define ORT_API_VERSION \([0-9][0-9]*\).*/\1/p' "$header" | head -1)"
[ -n "$served" ] || { echo "check-ort-api: no ORT_API_VERSION in $header" >&2; exit 2; }

if command -v cargo >/dev/null 2>&1 && [ -f "$here/../Cargo.lock" ]; then
  if asked=$(cd "$here/.." && cargo metadata --format-version 1 --locked 2>/dev/null); then
    from_features=$(printf '%s' "$asked" \
      | jq -r '[.resolve.nodes[] | select(.id | test("ort-sys")) | .features[] | select(startswith("api-")) | ltrimstr("api-") | tonumber] | max // empty' 2>/dev/null || true)
    if [ -n "${from_features:-}" ] && [ "$from_features" != "$required" ]; then
      echo "::error::scripts/native-libs.env pins ORT_REQUIRED_API=$required, but the resolved ort-sys asks for api-$from_features: the dependency changed its api-N feature, update the pin."
      exit 1
    fi
  fi
fi

echo "check-ort-api: ort asks for API ${required}; ${dir##*/} serves up to ${served}"
if [ "$served" -lt "$required" ]; then
  echo "::error::ONNX Runtime in ${dir} serves API ${served}, ort needs ${required}: local embeddings would panic at start (\"Failed to initialize ORT API\"). Raise ORT_VERSION — and ORT_REQUIRED_API if fastembed changed — in scripts/native-libs.env."
  exit 1
fi
