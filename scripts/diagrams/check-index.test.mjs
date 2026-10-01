import { test } from 'node:test';
import assert from 'node:assert/strict';
import { globToRegExp, parseIndex, parseHeader, checkDiagramFile, sharedOwnership, renderOrphans } from './check-index.mjs';

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

test('sharedOwnership: un fichier a un seul proprietaire', () => {
  const owners = new Map([
    ['backend:src/a.rs', ['po-x']],
    ['backend:src/b.rs', ['po-x', 'po-y']],
    ['backend:src/c.rs', ['po-z', 'po-z']], // meme diagramme via deux globs : pas un conflit
  ]);
  assert.deepEqual(sharedOwnership(owners), [{ file: 'backend:src/b.rs', names: ['po-x', 'po-y'] }]);
});

test('sharedOwnership: trie par chemin et dedoublonne les noms', () => {
  const owners = new Map([
    ['backend:src/z.rs', ['po-b', 'po-a', 'po-a']],
    ['backend:src/a.rs', ['po-c', 'po-d']],
  ]);
  assert.deepEqual(sharedOwnership(owners).map((s) => s.file), ['backend:src/a.rs', 'backend:src/z.rs']);
  assert.deepEqual(sharedOwnership(owners)[1].names, ['po-b', 'po-a']);
});

test('renderOrphans: compte, pourcentage et regroupement par depot puis dossier', () => {
  const md = renderOrphans(['frontend:src/x/b.ts', 'backend:src/a.rs', 'backend:src/x/c.rs'], 10);
  assert.match(md, /\*\*3 orphelins sur 10 fichiers source \(30\.0 %\)\.\*\*/);
  assert.match(md, /## backend \(2\)/);
  assert.match(md, /## frontend \(1\)/);
  assert.ok(md.indexOf('## backend') < md.indexOf('## frontend'), 'depots tries');
  assert.match(md, /- `src\/x\/` \(1\) : `c\.rs`/);
  assert.match(md, /- `src\/` \(1\) : `a\.rs`/);
  // un fichier a la racine du depot tombe dans le groupe "."
  assert.match(renderOrphans(['backend:build.rs'], 1), /- `\.\/` \(1\) : `build\.rs`/);
  assert.match(md, /--write-orphans/);
});

test('renderOrphans: deterministe (aucun sha ni date), donc verifiable hors reseau', () => {
  const a = renderOrphans(['backend:src/a.rs'], 2);
  const b = renderOrphans(['backend:src/a.rs'], 2);
  assert.equal(a, b);
  assert.doesNotMatch(a, /\d{4}-\d{2}-\d{2}/);
});

test('renderOrphans: zero orphelin ne divise pas par zero', () => {
  assert.match(renderOrphans([], 0), /\*\*0 orphelins sur 0 fichiers source \(0\.0 %\)\.\*\*/);
});
