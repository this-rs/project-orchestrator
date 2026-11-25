# Release: whole, or a draft

A release is published **complete**, or it is published as a **draft**. There is no third
outcome, and no release goes out silently partial.

This page says what "complete" means, where each guard sits, and how to repair a release that
came out incomplete. It is short on purpose: the guards live in `.github/workflows/release.yml`
and each one carries the measurement that put it there.

## Why this page exists

Measured on v0.0.20 (run `45638973782`, 25/11/2026, the 21st version):

- the three macOS `build-tauri` jobs failed — Apple answered **403 "a required agreement is
  missing or has expired"** to `notarytool`, and the workflow refuses to ship an unnotarized app,
  so the upload step was skipped and no `.dmg` reached the release;
- the `release` job had `if: always()`, so it published v0.0.20 anyway — tagged **Latest** —
  with no macOS installer at all;
- its generated `latest.json` therefore had **empty** `darwin-aarch64` and `darwin-x86_64`
  entries, so every Mac on 0.0.19 was offered an update it could not download;
- and the two Docker jobs sat **4 h 11** in `docker/build-push-action` (QEMU-emulated arm64,
  compiling `nexus-tools`, which links V8) and were still running, which refuses every
  `gh run rerun --failed` with `403 "the workflow run containing this job is already running"`.

Four independent holes. Each now has a guard.

## The promise

`Check the release is complete` compares `release-files/` against this list. A file added to the
promise must be added there; a file that stops being produced must be removed from both.

| Platform | Promised files |
|---|---|
| macOS desktop | `Project Orchestrator_<v>_aarch64.dmg`, `_x64.dmg`, `_x64-legacy.dmg` |
| macOS updater | `_arm64.app.tar.gz` + `.sig`, `_x86_64.app.tar.gz` + `.sig`, `_x86_64-legacy.app.tar.gz` + `.sig` |
| Linux desktop | `_amd64.AppImage`, `_amd64.deb`, `_arm64.deb`, `-<v>-1.x86_64.rpm`, `-<v>-1.aarch64.rpm` |
| Windows desktop | `_x64-setup.exe`, `_x64_en-US.msi` |
| Server, light and full | `orchestrator-[full-]<v>-{linux-x86_64,linux-arm64,macos-arm64,macos-x86_64,macos-x86_64-legacy,windows-x86_64}.{tar.gz,zip}` |
| Server packages | `project-orchestrator_<v>-1_{amd64,arm64}.deb`, `project-orchestrator-<v>-1.{x86_64,aarch64}.rpm` |
| Updater feed | `latest.json` with non-empty `darwin-aarch64` and `darwin-x86_64` (`url` **and** `signature`) |

**Known gap, not a blocker:** `latest.json` still has empty `linux-x86_64` and `windows-x86_64`
entries, because Tauri produces no updater bundle for the AppImage/NSIS targets today. The gate
prints a warning for those two instead of failing, so the gap is visible without blocking a
release. Closing it is its own task.

## What happens when something is missing

1. `Check the release is complete` writes the list to `missing.txt` and to the **job summary**,
   and sets the `complete` output.
2. The release is created with `draft: true`. A draft is never `/releases/latest`, and
   `install.sh`, the Homebrew formula, the website and the in-app updater all read
   `/releases/latest` — so nothing points a user at a file that does not exist.
3. The `Verdict — whole, or a draft` step **fails the job**. The run is red, and `update-homebrew`
   (which `needs: [release]`) is skipped, so the tap keeps the last good formula.

Repair: re-run the failed jobs, then re-run the `release` job. Note what was measured on
v0.0.20: re-running a **single** job creates a new *attempt* of the run, and that attempt's graph
does not always contain the `release` job. If it is absent, the two honest options are
`gh release upload` of the missing files onto the existing release, or a fresh
`workflow_dispatch` run for the tag (which rebuilds everything and re-creates the release).

## macOS: the Apple gate, checked before the build

Notarization is the one step no code can work around: Apple answers `403` to every submission
while an agreement is missing or expired, and only a human can accept it at
[developer.apple.com/account](https://developer.apple.com/account) → *Agreements*.

So `Notarization preflight (macOS)` runs `xcrun notarytool history` **before** the forty-minute
bundle build. On success it prints the accepted credentials and gets out of the way; on `403` it
fails the job in about thirty seconds with that URL and the words "Nothing was built".

The notarization step itself still treats any outcome other than `status: Accepted` as fatal:
a signed-but-unnotarized `.app` is rejected by Gatekeeper on every Mac, and shipping one is worse
than failing the release.

## Docker: one native build per architecture

`docker-build` is a matrix of `{full, api-only} × {amd64, arm64}`; each entry runs on a runner of
its own architecture (`ubuntu-22.04` / `ubuntu-22.04-arm`, the label the Linux arm64 desktop job
already uses), pushes **by digest**, and uploads that digest. `docker-manifest` collects the two
digests of a variant and writes the multi-arch manifest with `docker buildx imagetools create`.

No QEMU is involved. Both docker jobs carry `timeout-minutes`, because a job that never ends does
not merely fail: it holds the whole run in `in_progress`, and GitHub refuses `rerun --failed`
(403) for as long as it is in flight.

Caches are keyed per (variant, arch) — `release-docker-<variant>-<arch>` — so one architecture
never evicts the other's layers.

## Native libraries

ONNX Runtime and friends come from versioned `lib/<lib>-<version>` releases. That machinery, its
pins and its three-source resolution order are described in
[release-libraries.md](release-libraries.md).
