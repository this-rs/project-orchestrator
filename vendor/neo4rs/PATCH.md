# neo4rs 0.8.0 — local copy, negative one-byte integers read with their sign

This directory is **the crates.io crate `neo4rs 0.8.0` plus one upstream commit,
[`15ddd40f` "Fix tiny int handling" (neo4j-labs/neo4rs#252)][fix], applied
verbatim** (and one lint attribute, see "What changed"). It is wired in by the `[patch.crates-io]` block at the bottom of the
root `Cargo.toml`. It exists for one reason, and it should disappear as soon as
that reason is gone.

## Why

`BoltInteger::parse` (`src/types/integer.rs`) returned the marker byte of a Bolt
TINY_INT with `marker as i64`. The marker is a `u8`, so the negative one-byte
integers lost their sign: **every integer in -16..=-1 was read 240..=255**.

It is the only integer decoder of 0.8.0 (`BoltType::parse` in `src/types.rs`
calls it for every value; the serde "packstream" decoder only exists from 0.9
on), so the defect reached an integer alone in a column, inside a list, inside
a map, a node or relationship property, and the integer fields of dates and
durations. Encoding (`write_into`, same file) was correct: a `-1` sent by the
backend is stored `-1`. The crate's own tests had pinned the defect — they
expected `255` for the `-1` of a `PULL` message.

What it did here. `-1` is the backend's sentinel for "none" / "not computed".
On a copy of the production graph (2026-11-11) the stored values in -16..=-1
were 31 858, all of them `-1`: `File.wl_hash_version` (20 607),
`File.structural_dna_version` (11 162), `ProtocolState.state_timeout_secs` (89).
The first two have no Rust reader. The third has one
(`Neo4jClient::node_to_protocol_state`): a state without timeout was read with
a **255-second timeout** instead of the 300 s / 1 800 s defaults of the
runner, and `protocol::seed::seed_prompt_fragments` — read the states, write
them back — stored that 255 for good (65 states on the same copy).

**The 255 already written, and the trap that remains.** The data repair is code,
not a manual query: `Neo4jClient::repair_phantom_state_timeouts`
(`src/neo4j/data_migrations.rs`) resets every `ProtocolState.state_timeout_secs = 255`
to `-1` at each start of the fixed binary. It is replayed on EVERY start, not marked
complete, because of an OLD binary (one without this fix) still connected to the same
database: it reads `-1` as 255 and writes it back at its next `seed_prompt_fragments`.
So: **every old binary must be stopped** (service, Docker container, desktop app on the
same database) — the replay repairs what one wrote, it cannot stop it writing. If it
logs "an older binary is still connected" start after start, one is. The replay costs a
label scan of `ProtocolState` (hundreds of nodes) and writes nothing when there is
nothing to repair. A 255 someone CHOOSES is safe from it: the write path marks a
positive timeout written on a state that did not hold it (`state_timeout_explicit`),
a value merely read and written back earns no marker, and the repair touches only a
255 without the marker. Today no write path sets `Some(n)` at all (13 constructors at
`None`, `PortableState` does not carry the field, no API body deserialises a
`ProtocolState`); the day one does, it is already covered.

There is no upstream release to move to without changing the backend:

| neo4rs | carries #252 | note |
|---|---|---|
| 0.7.x, 0.8.0 (ours, latest stable) | no | |
| 0.9.0-rc.2 … rc.6 | no | |
| 0.9.0-rc.7 … rc.11 (latest published) | yes | release candidates; public API differs from 0.8 |

The backend has about 2 000 `neo4rs` decoding sites. Moving them to a
release-candidate API is its own change, with its own review; it is not the
way to fix a wrong sign. And a git fork pinned by `rev` (the way `nexus-claude`
is pinned) needs a repository that must exist and stay reachable for every
build, plus an `allow-git` entry in `deny.toml`; a copy in the tree needs
neither and is reviewed next to the reason for it.

## What changed

Reproduce the copy and diff it:

```sh
curl -sSLo /tmp/neo4rs-0.8.0.crate \
  https://static.crates.io/crates/neo4rs/neo4rs-0.8.0.crate
tar xzf /tmp/neo4rs-0.8.0.crate -C /tmp
diff -ru /tmp/neo4rs-0.8.0/src vendor/neo4rs/src
diff -ru /tmp/neo4rs-0.8.0/include vendor/neo4rs/include
```

Expected output — the three files of the upstream commit, one local line, and
nothing in `include/`:

1. `src/types/integer.rs`: `marker as i64` → `(marker as i8) as i64` (the fix),
   and three assertions added to the crate's `should_deserialize_integer`
   (`0x00` → 0, `0xF0` → -16, `0x7F` → 127).
2. `src/messages/pull.rs` and `src/messages/discard.rs`: the crate's tests
   expected `255` for `n` and `qid`; they now expect `-1`.
3. `src/lib.rs`, local and not upstream: `#![allow(warnings, clippy::all)]` at
   the top. Cargo builds a registry crate with `--cap-lints allow` and a path
   crate without it, so the 16 warnings of 0.8.0 on today's toolchain
   (deprecated chrono calls, dead code) would otherwise appear in every build
   of the backend. It changes no generated code.

Not vendored, because nothing builds them from here: `tests/`, `examples/`,
`Makefile.toml`, `[dev-dependencies]` and the `[[test]]` / `[[example]]`
sections of `Cargo.toml` that need them. `[features]` and every dependency stay
exactly as published. `include/` stays: `src/lib.rs` embeds it in its
documentation with `include_str!`.

Licence: MIT (the `license` field of `Cargo.toml`; the published crate ships no
licence file).

One thing the copy costs: `cargo audit` and `cargo deny check advisories` match
advisories against registry packages. An advisory published against `neo4rs`
itself would not be reported for this path copy. Its dependencies are still
checked.

## Verification

```sh
grep -A2 '^name = "neo4rs"$' Cargo.lock   # version 0.8.0, no `source` line
cargo tree -i neo4rs                      # one copy, at vendor/neo4rs
NEO4J_URI=bolt://localhost:17687 NEO4J_PASSWORD=testpassword HARNESS_NEO4J_REQUIRED=1 \
  cargo test --test neo4j_integer_fidelity   # a throwaway Neo4j, never a live one
```

`tests/neo4j_integer_fidelity.rs` is the proof: against the published 0.8.0 its
driver tests fail (the integers -16..=-1 come back 240..=255, a state without
timeout is read `Some(255)`); against this copy they pass. It runs in the
`integration-tests` and `coverage` jobs of the CI.

## How to remove it

When the backend moves to a published `neo4rs` that contains #252 (0.9.0-rc.7
or later, in practice the 0.9.0 release):

1. delete `vendor/neo4rs/`,
2. delete the `[patch.crates-io]` block and the `"vendor/neo4rs"` entry of
   `exclude` in the root `Cargo.toml`, and the comment above `neo4rs = "0.8"`,
3. drop the `COPY vendor ./vendor/` line of the `Dockerfile` and restore the
   first line of `.cargo/config.toml`,
4. bump `neo4rs` (the API changes are the real work of that step),
5. keep `tests/neo4j_integer_fidelity.rs`: it is what says whether the next
   driver reads integers correctly.

[fix]: https://github.com/neo4j-labs/neo4rs/commit/15ddd40f0203963791295e5a0d893e56b4610708
