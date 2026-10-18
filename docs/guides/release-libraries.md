# Release: prebuilt native libraries and durable caches

A release must not recompile what did not change. This page describes the two mechanisms that
enforce it, where their inputs live, and what to do when one of them is missing.

## Why

Measured on the v0.0.17 release run: the Docker jobs took **214 min** (api-only) and **122 min**
(full), while every other job sat between 13 and 25 min. On top of that, ONNX Runtime for macOS
Intel had to be **built** (≈ 1 h) by nearly every release, because the cache it relied on was
`actions/cache` — invisible to a tag-triggered run, and evicted by GitHub after 7 idle days.

## 1. Versioned native libraries

| What | Where |
|---|---|
| The pins — the **only** place a version is written | `scripts/native-libs.env` |
| The resolver (`use` / `fetch` / `publish` / `build` / `mirror` / `plan-*`) | `scripts/prebuilt-lib.sh` |
| The producer | `.github/workflows/prebuilt-libs.yml` |
| The consumers | `.github/workflows/release.yml` (`build-backend`, `build-tauri`) |

Artifacts are published as assets of a **library release**, tagged `lib/<lib>-<version>` and marked
**prerelease**. That matters twice over: a prerelease can never become `/releases/latest` (which
`install.sh` and the Homebrew formula read), and a `lib/...` tag does not match the `v*` trigger of
`release.yml`.

`prebuilt-lib.sh` resolves in this order, and logs which source won:

1. the library release of this repository — durable, versioned, readable by a tag-triggered run;
2. the upstream archive (Microsoft, for ONNX Runtime);
3. a local build — **only** for a `(lib, target)` upstream stopped publishing (macOS Intel for `ort`
   from 1.24), and **only** where the host is the target: a cross-built native library is the wrong
   architecture at runtime.

Whatever the source, `scripts/check-ort-api.sh` runs on the result: a runtime that does not serve
the API `ort` was compiled for makes local embeddings panic at start. That check caught a defect
already shipped in v0.0.15–v0.0.17 (arm64 Linux and macOS Intel carried ORT 1.23.2, which serves
API 23, while `ort` asks for 24).

### Bumping a library

1. Edit the pin in `scripts/native-libs.env`.
2. If the pin makes the library unsupported at runtime, `check-ort-api.sh` fails the release before
   anything is packaged — fix the pin, not the check.
3. Merge to `main`. `prebuilt-libs.yml` runs on that path change, builds or mirrors what is missing,
   and publishes it. Re-running it is harmless: an existing asset is left alone.
4. Nothing else to change. `release.yml` reads the same file.

### When a release runs while the artifact is missing

It falls back (upstream, then a local build) and warns. That is deliberate — a release must not be
blocked by a missing artifact — but it is also a signal: re-run **Prebuilt native libraries** so the
next release starts warm.

## 2. Durable Docker layer cache

The Docker jobs use two caches, both keyed by variant (before, the two variants shared one scope and
evicted each other under a 10 GB cap):

- `type=gha` — the fast path within a branch;
- `type=registry,ref=ghcr.io/<repo>-cache:docker-<variant>` — a cache **image**, with no size cap and
  no 7-day eviction, which is what makes a release start warm months later.

The cache image is a package, not a published product: it holds no runnable image, only layers.

## What is deliberately not in the channel

- **`nexus-tools`** (and the V8 snapshot it embeds) is keyed by the nexus revision `Cargo.toml` pins.
  That revision changes with nearly every release, so a prebuilt artifact would almost never be
  reused; the build already happens once per target and is shared between jobs by artifact.
- **The backend binaries themselves** are workspace builds: their cache is `swatinem/rust-cache`,
  which is the right tool for a source-keyed build.
