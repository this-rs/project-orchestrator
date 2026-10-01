import { test } from 'node:test';
import assert from 'node:assert/strict';
import { globToRegExp, parseIndex, parseHeader, checkDiagramFile } from './check-index.mjs';

test('glob: **, *, braces', () => {
  assert.ok(globToRegExp('src/heartbeat/**').test('src/heartbeat/checks/git_drift.rs'));
  assert.ok(globToRegExp('src/api/{mod,routes}.rs').test('src/api/routes.rs'));
  assert.ok(!globToRegExp('src/api/{mod,routes}.rs').test('src/api/handlers.rs'));
  assert.ok(globToRegExp('tests/*.rs').test('tests/api_tests.rs'));
  assert.ok(!globToRegExp('tests/*.rs').test('tests/x/y.rs'));
});

test('parseIndex: scalars and covers', () => {
  const [e] = parseIndex('diagrams:\n  - name: po-x\n    status: planned\n    covers:\n      - "backend:src/a.rs"\n      - "frontend:src/**"\n');
  assert.equal(e.name, 'po-x');
  assert.equal(e.status, 'planned');
  assert.deepEqual(e.covers, ['backend:src/a.rs', 'frontend:src/**']);
});

test('parseIndex rejects unknown lines', () => {
  assert.throws(() => parseIndex('diagrams:\n  - name: a\n  bogus line\n'));
});

const entry = { name: 'po-x', covers: ['backend:src/a.rs', 'backend:src/b/**'] };
const good = '%% name: po-x\n%% covers: backend:src/b/** backend:src/a.rs\n%% verified: 94c04bce\nflowchart TD\n';

test('header: valid file passes', () => {
  assert.deepEqual(parseHeader(good), { name: 'po-x', covers: 'backend:src/b/** backend:src/a.rs', verified: '94c04bce' });
  assert.deepEqual(checkDiagramFile(entry, good), []);
});

test('header: wrong name, covers drift and bad sha are reported', () => {
  assert.equal(checkDiagramFile(entry, good.replace('po-x', 'po-y')).length, 1);
  assert.equal(checkDiagramFile(entry, good.replace('backend:src/a.rs', 'backend:src/c.rs')).length, 1);
  assert.equal(checkDiagramFile(entry, good.replace('94c04bce', 'HEAD')).length, 1);
  assert.equal(checkDiagramFile(entry, 'flowchart TD\n').length, 3);
});
