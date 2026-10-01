import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { lintLocal, negativeControl, MERMAID_PATH } from './publish.mjs';

const fx = (n) => readFileSync(new URL(`./fixtures/${n}`, import.meta.url), 'utf8');

before(async () => {
  if (!existsSync(MERMAID_PATH)) {
    const r = await fetch('https://mermaid.ffs.dev/vendor/mermaid.min.js');
    assert.ok(r.ok, 'cannot fetch mermaid.min.js');
    writeFileSync(MERMAID_PATH, Buffer.from(await r.arrayBuffer()));
  }
});

test('negative control rejects a broken diagram', () => negativeControl());
test('valid diagram passes', async () => assert.equal((await lintLocal(fx('valid.mmd'))).ok, true));
test('broken diagram fails', async () => assert.equal((await lintLocal(fx('broken.mmd'))).ok, false));
test('bare %% is refused', async () => {
  const r = await lintLocal(fx('bare-comment.mmd'));
  assert.equal(r.ok, false); assert.match(r.errors[0], /bare/);
});
test('state "x" as Y is refused', async () => {
  const r = await lintLocal(fx('state-as.mmd'));
  assert.equal(r.ok, false); assert.match(r.errors[0], /state/);
});
