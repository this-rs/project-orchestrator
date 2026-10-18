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
# Needs: cargo, jq (the required version is read from the resolved features, not written here).
set -euo pipefail

dir="${1:?usage: check-ort-api.sh <onnxruntime dir>}"
header="$dir/include/onnxruntime_c_api.h"
[ -f "$header" ] || { echo "check-ort-api: no $header" >&2; exit 2; }

served="$(sed -n 's/^#define ORT_API_VERSION \([0-9][0-9]*\).*/\1/p' "$header" | head -1)"
[ -n "$served" ] || { echo "check-ort-api: no ORT_API_VERSION in $header" >&2; exit 2; }

# The resolved features of ort-sys: the highest api-N is what `ort` asks for (17 without any).
required="$(cargo metadata --format-version 1 --locked \
  | jq -r '[.resolve.nodes[] | select(.id | test("ort-sys")) | .features[] | select(startswith("api-")) | ltrimstr("api-") | tonumber] | max // 17')"

echo "check-ort-api: ort asks for API ${required}; ${dir##*/} serves up to ${served}"
if [ "$served" -lt "$required" ]; then
  echo "::error::ONNX Runtime in ${dir} serves API ${served}, ort needs ${required}: local embeddings would panic at start (\"Failed to initialize ORT API\"). Raise ORT_VERSION in .github/workflows/release.yml."
  exit 1
fi
