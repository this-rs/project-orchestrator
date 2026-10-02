#!/usr/bin/env node
// Fails when a forbidden token appears anywhere in the tracked tree.
//
// WHY THIS EXISTS. This repository is public. A few identifiers — a private
// service host, its API paths — must never appear in it, in code, docs,
// scripts, CI or fixtures. The rule was previously honoured by hand, and it
// was not honoured: a token was found on fifteen published branches while
// `main` happened to be clean.
//
// WHY IT STORES HASHES AND NOT THE STRINGS. A gate that carries the string it
// forbids publishes it. `forbidden-digests.json` therefore holds only the
// SHA-256 of each token's normalized form, never the token. The consequence,
// accepted deliberately: this checker can tell you a forbidden token is at
// `src/foo.rs:42` but cannot print it. You read the line.
//
// WHAT IT CATCHES. Normalization strips case and every separator, so one
// digest covers `a.b.c`, `A-B-C`, `a_b_c` and `a b c`. Candidates are built
// from single tokens and from windows of up to three consecutive tokens, which
// is what makes the spaced-out prose form reachable.
//
// WHAT IT DOES NOT CATCH, and you should not believe otherwise:
//   * commit messages — only the tree is read. A forbidden token in a commit
//     message passes this gate. Checking them needs full history, which the
//     default CI checkout (depth 1) does not fetch.
//   * a token split across a NEWLINE. Candidates are built per line, so a
//     token broken over two lines passes. Punctuation inside a line does not
//     help an evader: the normalizer strips it, which is the point.
//   * a token interrupted by a letter or a digit, since those survive
//     normalization and change the candidate.
//   * a token reached through more than three whitespace-separated words.
//   * anything encoded — base64, percent-escapes, \u sequences.
//
// This file deliberately contains no example of a forbidden form. Writing one
// out — even obfuscated with slashes or stars — reintroduces the string, since
// normalization is exactly what removes that obfuscation. The first version of
// this comment did, and the gate caught its own source file. Kept as a warning
// rather than silently fixed: the tests below pin both behaviours instead.
// It is a ratchet against the accident, not a defence against someone who
// wants the string in. That is the right trade for a gate that must run
// offline: measured at ~2.1 s over 612 tracked files on this repository.

import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { readFileSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))

/** Lowercase and drop everything that is not a letter or a digit. */
export const normalize = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, '')

const sha256 = (s) => createHash('sha256').update(s).digest('hex')

/** Tokens, as the coarse runs of characters an identifier or URL is made of. */
const TOKEN = /[A-Za-z0-9][A-Za-z0-9._\-]*/g

/** How many consecutive tokens to join when looking for a spaced-out form. */
const MAX_WINDOW = 3

// Skipped wholesale: binary, generated, or vendored. `dist/` is a build output
// and would otherwise make every run read megabytes of bundled JavaScript.
const SKIP_DIRS = ['dist/', 'target/', 'node_modules/', '.git/']
const SKIP_EXT = new Set([
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'ico', 'svgz', 'pdf', 'woff', 'woff2',
  'ttf', 'otf', 'eot', 'zip', 'gz', 'bz2', 'xz', 'zst', 'tar', 'bin', 'wasm',
  'so', 'dylib', 'dll', 'exe', 'a', 'o', 'rlib', 'pack', 'idx', 'lock',
  'mp3', 'mp4', 'mov', 'webm', 'avif', 'heic',
])

const MAX_BYTES = 2 * 1024 * 1024

export function loadDigests (path = join(HERE, 'forbidden-digests.json')) {
  const raw = JSON.parse(readFileSync(path, 'utf8'))
  const byLength = new Map()
  for (const e of raw.entries) {
    if (!Number.isInteger(e.len) || e.len < 4) {
      throw new Error(`forbidden-digests.json: bad len ${JSON.stringify(e.len)}`)
    }
    if (!/^[0-9a-f]{64}$/.test(e.sha256)) {
      throw new Error('forbidden-digests.json: sha256 must be 64 lowercase hex chars')
    }
    if (!byLength.has(e.len)) byLength.set(e.len, new Set())
    byLength.get(e.len).add(e.sha256)
  }
  if (byLength.size === 0) throw new Error('forbidden-digests.json: no entries')
  return byLength
}

/**
 * Line numbers of `text` that carry a forbidden token.
 *
 * Only candidates whose normalized length matches a digest length are hashed,
 * so the cost is roughly one hash per token per distinct forbidden length —
 * not one per substring.
 */
export function findInText (text, byLength) {
  const lengths = [...byLength.keys()]
  const hits = []
  const lines = text.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const tokens = lines[i].match(TOKEN)
    if (!tokens) continue
    const norms = tokens.map(normalize)

    for (let a = 0; a < norms.length; a++) {
      let joined = ''
      for (let w = 0; w < MAX_WINDOW && a + w < norms.length; w++) {
        joined += norms[a + w]
        // Longer windows only grow the candidate, so stop once past every length.
        if (joined.length > Math.max(...lengths)) break
        const digests = byLength.get(joined.length)
        if (digests && digests.has(sha256(joined))) {
          hits.push(i + 1)
          a = norms.length // this line is already reported; move on
          break
        }
      }
    }
  }
  return hits
}

const skipped = (p) =>
  SKIP_DIRS.some((d) => p.startsWith(d) || p.includes(`/${d}`)) ||
  SKIP_EXT.has(p.slice(p.lastIndexOf('.') + 1).toLowerCase())

function trackedFiles (cwd) {
  return execFileSync('git', ['ls-files', '-z'], { cwd, maxBuffer: 64 * 1024 * 1024 })
    .toString('utf8')
    .split('\0')
    .filter(Boolean)
}

export function scanRepo (cwd = process.cwd(), byLength = loadDigests()) {
  const findings = []
  let read = 0
  for (const rel of trackedFiles(cwd)) {
    if (skipped(rel)) continue
    const abs = join(cwd, rel)
    let text
    try {
      if (statSync(abs).size > MAX_BYTES) continue
      text = readFileSync(abs, 'utf8')
    } catch {
      continue // deleted between ls-files and read, or not valid UTF-8
    }
    if (text.includes('\0')) continue
    read++
    for (const line of findInText(text, byLength)) findings.push({ file: rel, line })
  }
  return { findings, read }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { findings, read } = scanRepo(process.cwd())
  if (findings.length === 0) {
    console.log(`${read} tracked text files scanned — no forbidden token.`)
    console.log('Scope: the tree only. Commit messages are NOT checked (see the comment at the top).')
    process.exit(0)
  }
  console.error(`${findings.length} forbidden token occurrence(s) in ${read} scanned files.`)
  console.error('The token itself is not printed: this gate stores hashes, not strings.')
  console.error('Open each location and remove what you find there.\n')
  for (const f of findings) console.error(`  ${f.file}:${f.line}`)
  process.exit(1)
}
