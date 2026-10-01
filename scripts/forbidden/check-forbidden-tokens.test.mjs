// Run with: node --test scripts/forbidden/check-forbidden-tokens.test.mjs
//
// These tests never write a real forbidden token. The positive cases build
// their own digest set from a harmless synthetic token, which exercises exactly
// the same matching path; the negative cases run against the REAL digest file,
// which is what proves the gate does not fire on the legitimate vocabulary it
// sits next to.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { normalize, findInText, loadDigests } from './check-forbidden-tokens.mjs'

/** A digest set for an arbitrary token, built the way the real file was. */
const digestsFor = (...tokens) => {
  const m = new Map()
  for (const t of tokens) {
    const n = normalize(t)
    if (!m.has(n.length)) m.set(n.length, new Set())
    m.get(n.length).add(createHash('sha256').update(n).digest('hex'))
  }
  return m
}

const SAMPLE = 'widget.example.invalid' // normalizes to widgetexampleinvalid
const SET = digestsFor(SAMPLE)

test('normalize drops case and every separator', () => {
  assert.equal(normalize('A.B-C_D E'), 'abcde')
  assert.equal(normalize('Widget.Example.Invalid'), 'widgetexampleinvalid')
})

test('catches the plain form', () => {
  assert.deepEqual(findInText(`host = "${SAMPLE}"`, SET), [1])
})

test('one digest covers every separator and case variant', () => {
  for (const v of [
    'widget.example.invalid',
    'WIDGET.EXAMPLE.INVALID',
    'widget-example-invalid',
    'widget_example_invalid',
    'Widget.Example-Invalid',
  ]) {
    assert.deepEqual(findInText(`x ${v} y`, SET), [1], `missed variant ${v}`)
  }
})

test('catches it inside a URL, where separators are punctuation', () => {
  assert.deepEqual(findInText('fetch("https://widget.example.invalid/api/v1/x")', SET), [1])
})

test('catches the spaced-out prose form, up to three words', () => {
  assert.deepEqual(findInText('the widget example invalid service', SET), [1])
})

test('does NOT reach past a three-word window — a documented blind spot', () => {
  // Four words normalize to the same string, and the gate misses it on purpose:
  // widening the window costs hashes on every token of every file.
  const SET4 = digestsFor('widget example invalid service')
  assert.deepEqual(findInText('widget example invalid service', SET4), [])
})

test('reports the line number, and a line only once', () => {
  const text = ['clean', `a ${SAMPLE} b`, 'clean', `${SAMPLE} ${SAMPLE}`].join('\n')
  assert.deepEqual(findInText(text, SET), [2, 4])
})

test('a longer token containing the forbidden one is not a match', () => {
  // Matching is on the whole normalized candidate, so a substring does not fire.
  assert.deepEqual(findInText('xwidget.example.invalidx', SET), [])
})

test('the real digest file loads and holds at least one entry', () => {
  const real = loadDigests()
  assert.ok(real.size >= 1)
  for (const [len, set] of real) {
    assert.ok(Number.isInteger(len) && len >= 4)
    for (const d of set) assert.match(d, /^[0-9a-f]{64}$/)
  }
})

test('the real digests do NOT fire on the legitimate vocabulary', () => {
  // This repository documents Mermaid diagrams and must keep being able to.
  // If this test ever fails, a digest was added that forbids an ordinary word.
  const real = loadDigests()
  for (const ok of [
    'mermaid',
    'Mermaid diagram',
    'docs/diagrams/nexus-sdk-types.mmd',
    'see the Mermaid charter in docs/diagrams/README.md',
    'ffs',
    'https://auth.example.com/oauth/token',
  ]) {
    assert.deepEqual(findInText(ok, real), [], `false positive on: ${ok}`)
  }
})

test('loadDigests refuses a malformed file rather than passing silently', async () => {
  const { writeFileSync, mkdtempSync } = await import('node:fs')
  const { join } = await import('node:path')
  const { tmpdir } = await import('node:os')
  const dir = mkdtempSync(join(tmpdir(), 'fbd-'))

  const bad = [
    { entries: [] },
    { entries: [{ len: 2, sha256: 'a'.repeat(64) }] },
    { entries: [{ len: 13, sha256: 'NOTHEX' }] },
    { entries: [{ len: '13', sha256: 'a'.repeat(64) }] },
  ]
  for (const [i, b] of bad.entries()) {
    const p = join(dir, `bad${i}.json`)
    writeFileSync(p, JSON.stringify(b))
    assert.throws(() => loadDigests(p), undefined, `should have rejected ${JSON.stringify(b)}`)
  }
})
