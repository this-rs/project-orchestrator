# candle-core 0.8.4 — local copy, zip 9 only

This directory is **the crates.io crate `candle-core 0.8.4`, unmodified except
for the `zip` requirement and the three call sites that requirement forces.**
It is wired in by the `[patch.crates-io]` block at the bottom of the root
`Cargo.toml`. It exists for one reason, and it should disappear as soon as that
reason is gone.

## Why

`candle-core 0.8.4` depends on `zip ^1.1` (through its numpy `npz` and PyTorch
`.pth` loaders). That copy — `zip 1.1.4` in `Cargo.lock` — matches two GitHub
advisories whose only fixed versions are 9.0.0 and 9.0.5:

| dependabot | advisory | severity | fixed in |
|---|---|---|---|
| #47 | GHSA-m665-v7pm-7grj — parser differential between `ZipArchive` and the streaming reader (dual-header archives) | critical | `zip >= 9.0.5` |
| #49 | GHSA-8hxf-qr8p-62x4 — AES entry HMAC bypass on premature EOF | moderate | `zip >= 9.0.0` |

Everything else we declare already resolves `zip 9.1.3`: the root crate (self-update
reader, document export), `calamine 0.37`, `office_oxide 0.1.16`,
`tauri-plugin-updater 2.14.1`. `candle-core` was the only consumer left behind.

There is no upstream release to move to. Requirement by release:

| candle-core | `zip` requirement |
|---|---|
| 0.8.4 (ours) | `^1.1` |
| 0.9.x | `^1.1` |
| 0.10.x | `^7.2` |
| 0.11.0 (latest published) | `^8.6` |
| `main` (unreleased) | `9.1.3` |

And cargo cannot widen a transitive requirement from the outside: a
`[patch.crates-io]` entry is **ignored** when the patched version does not
satisfy the dependent's own requirement, so patching `zip` to 9.1.3 does nothing
as long as `candle-core` asks for `^8.6`.

## What changed

Reproduce the copy and diff it:

```sh
curl -sSLo /tmp/candle-core-0.8.4.crate \
  https://static.crates.io/crates/candle-core/candle-core-0.8.4.crate
tar xzf /tmp/candle-core-0.8.4.crate -C /tmp
diff -ru /tmp/candle-core-0.8.4/src vendor/candle-core/src
```

Expected output — three hunks, all of them copies of what upstream did on
`main` when it moved to `zip 9.1.3`:

1. `Cargo.toml`: `zip` `1.1.1` → `9.1.3`, `default-features = false` unchanged.
2. `src/npy.rs` (twice): `ZipFile::name()` returns `ZipResult<Cow<'_, str>>` in
   zip 9, so `reader.name()` became `reader.name()?` and the `unwrap_or` borrows
   `&name`.
3. `src/pickle.rs` (once): `file_names()` now yields `ZipResult<Cow<'_, str>>`,
   so the names are collected through `Result` instead of `to_string()`.

Not vendored, because nothing builds them from here: `tests/`, `examples/`,
`benches/`, `[dev-dependencies]` and the `[[test]]` / `[[example]]` / `[[bench]]`
sections they need. `[features]` and every other dependency stay exactly as
published — `cuda`, `metal`, `mkl` and `accelerate` included, still optional and
still off.

Licence: MIT OR Apache-2.0, `LICENSE` kept next to the sources.

## Verification

```sh
cargo tree -i zip            # only zip v9.1.3, one copy
grep -c '^name = "zip"' Cargo.lock   # 1
cargo check -p neural-routing-gnn    # the only consumer of candle in this repo
```

`cargo metadata --locked` (what `security.yml` runs) also has to stay quiet: a
patch that the lock does not agree with is the exact failure the `zip 9.0.5`
bump hit in #1017.

## How to remove it

When a candle release carries `zip >= 9.0.5`:

1. delete `vendor/candle-core/`,
2. delete the `[patch.crates-io]` block and the `"vendor/candle-core"` entry of
   `exclude` in the root `Cargo.toml`,
3. bump `candle-core` / `candle-nn` there (the version that carries zip 9 may
   need code changes in `crates/neural-routing-*`),
4. drop the `COPY vendor ./vendor/` line of the `Dockerfile`,
5. `cargo update -p candle-core`, then `cargo tree -i zip`.
