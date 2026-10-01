import { test } from 'node:test';
import assert from 'node:assert/strict';
import { globToRegExp, parseIndex, parseHeader, checkDiagramFile, sharedOwnership, renderOrphans, localIndexOwners, orphanCeiling } from './check-index.mjs';

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

const nexusFiles = {
  nexus: ['claude-code-api/src/core/model_registry.rs', 'claude-code-api/src/main.rs', 'claude-code-sdk-rs/src/lib.rs'],
  frontend: ['src/App.tsx'],
};
const nexusIndex = [{
  repo: 'nexus',
  entries: parseIndex('diagrams:\n  - name: nexus-model-catalogue\n    status: verified\n    covers:\n      - "nexus:claude-code-api/src/core/model_registry.rs"\n'),
}];

test('localIndexOwners: un index local possede les fichiers de son depot', () => {
  const owned = localIndexOwners(nexusIndex, nexusFiles);
  assert.deepEqual([...owned.keys()], ['nexus:claude-code-api/src/core/model_registry.rs']);
  assert.equal(owned.get('nexus:claude-code-api/src/core/model_registry.rs'), 'nexus/INDEX.yml#nexus-model-catalogue');
});

test('localIndexOwners: un index local ne peut pas revendiquer un AUTRE depot', () => {
  // Sinon nexus pourrait s'attribuer du code du frontend sans que personne la-bas le sache.
  const greedy = [{ repo: 'nexus', entries: parseIndex('diagrams:\n  - name: nexus-x\n    status: verified\n    covers:\n      - "frontend:src/App.tsx"\n      - "nexus:claude-code-api/src/main.rs"\n') }];
  const owned = localIndexOwners(greedy, nexusFiles);
  assert.deepEqual([...owned.keys()], ['nexus:claude-code-api/src/main.rs']);
});

test('localIndexOwners: globs et accolades resolus comme dans l index principal', () => {
  const idx = [{ repo: 'nexus', entries: parseIndex('diagrams:\n  - name: nexus-y\n    status: verified\n    covers:\n      - "nexus:claude-code-api/src/**"\n') }];
  assert.equal(localIndexOwners(idx, nexusFiles).size, 2);
});

test('localIndexOwners: entrees sans covers, ou depot absent, ne cassent rien', () => {
  assert.equal(localIndexOwners([{ repo: 'nexus', entries: [{ name: 'x', status: 'verified', covers: [] }] }], nexusFiles).size, 0);
  assert.equal(localIndexOwners(nexusIndex, {}).size, 0);
  assert.equal(localIndexOwners([], nexusFiles).size, 0);
});

test('renderOrphans: les index locaux sont expliques dans le registre publie', () => {
  const md = renderOrphans(['backend:src/a.rs'], 10, [{ repo: 'nexus', diagrams: 3, owned: 24 }]);
  assert.match(md, /- `nexus` : 3 diagrammes, 24 fichiers possedes la-bas/);
  assert.match(md, /collision entre les deux index est une erreur/);
  // sans index local, pas de section parasite
  assert.doesNotMatch(renderOrphans(['backend:src/a.rs'], 10), /possedes la-bas/);
});

// --- controle negatif de bout en bout : le gate doit REFUSER une collision entre index.
// Un gate qui ne refuse rien est pire que pas de gate, parce qu'il se cite comme preuve.
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const script = join(dirname(fileURLToPath(import.meta.url)), 'check-index.mjs');

function fixture({ mainCovers, localCovers }) {
  const root = mkdtempSync(join(tmpdir(), 'diagram-index-'));
  const be = join(root, 'backend');
  const nx = join(root, 'nexus');
  mkdirSync(join(be, 'docs/diagrams'), { recursive: true });
  mkdirSync(join(be, 'src'), { recursive: true });
  mkdirSync(join(nx, 'claude-code-api/src/core'), { recursive: true });
  // frontend et website existent, vides : sans eux tout tourne en portee incomplete et le
  // cliquet ne s'applique jamais — c'est le piege que ces fixtures doivent eviter.
  mkdirSync(join(root, 'frontend/src'), { recursive: true });
  mkdirSync(join(root, 'website/src'), { recursive: true });
  writeFileSync(join(be, 'src/own.rs'), '// backend\n');
  writeFileSync(join(nx, 'claude-code-api/src/core/model_registry.rs'), '// nexus\n');
  writeFileSync(join(be, 'docs/diagrams/INDEX.yml'),
    `diagrams:\n  - name: po-carte\n    owner: t\n    status: planned\n    role: index\n    covers:\n      - "backend:docs/diagrams/INDEX.yml"\n  - name: po-x\n    owner: t\n    status: planned\n    covers:\n${mainCovers.map((c) => `      - "${c}"\n`).join('')}`);
  if (localCovers) {
    mkdirSync(join(nx, 'docs/diagrams'), { recursive: true });
    // verified : seule une entree verified possede un fichier (cf. la regle du plafond d'orphelins)
    writeFileSync(join(nx, 'docs/diagrams/INDEX.yml'),
      `diagrams:\n  - name: nexus-model-catalogue\n    owner: t\n    status: verified\n    covers:\n${localCovers.map((c) => `      - "${c}"\n`).join('')}`);
  }
  return { root, be, nx, fe: join(root, 'frontend'), web: join(root, 'website') };
}

function run(be, nx, args = []) {
  // stdout ET stderr : les avertissements (plafond releve, depot absent) passent par stderr,
  // et un test qui ne lit que stdout les manquerait.
  const r = spawnSync('node', [script, ...args], {
    encoding: 'utf8',
    env: {
      ...process.env,
      DIAGRAM_ROOT_BACKEND: be,
      DIAGRAM_ROOT_NEXUS: nx,
      DIAGRAM_ROOT_FRONTEND: join(be, '..', 'frontend'),
      DIAGRAM_ROOT_WEBSITE: join(be, '..', 'website'),
    },
  });
  return { code: r.status, out: `${r.stdout ?? ''}${r.stderr ?? ''}` };
}

test('gate: une collision entre index principal et index local est REFUSEE', () => {
  const { root, be, nx } = fixture({
    mainCovers: ['backend:src/own.rs', 'nexus:claude-code-api/src/core/model_registry.rs'],
    localCovers: ['nexus:claude-code-api/src/core/model_registry.rs'],
  });
  try {
    const { code, out } = run(be, nx);
    assert.equal(code, 1, 'le script doit echouer');
    assert.match(out, /revendique par po-x ET par nexus\/INDEX\.yml#nexus-model-catalogue/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: sans collision il passe, et le fichier de l index local n est pas orphelin', () => {
  const { root, be, nx } = fixture({
    mainCovers: ['backend:src/own.rs'],
    localCovers: ['nexus:claude-code-api/src/core/model_registry.rs'],
  });
  try {
    const { code, out } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 0, out);
    assert.match(out, /index local 'nexus' : 1 diagrammes, 1 fichiers possedes ailleurs/);
    // po-x est `planned` : son perimetre reste orphelin, seul le fichier de nexus est possede.
    assert.match(out, /1 orphelins/);
    assert.match(out, /dont 1 reserves par une entree planned/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: sans index local, le fichier du depot voisin est bien orphelin', () => {
  // Controle du controle : si localIndexOwners cessait de fonctionner, ce test passerait
  // et le precedent echouerait. Les deux ensemble pincent la logique.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    const { code, out } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 0, out);
    // sans index local : le fichier de nexus redevient orphelin, et po-x etant `planned`,
    // src/own.rs l'est aussi -> 2.
    assert.match(out, /2 orphelins/);
    assert.doesNotMatch(out, /index local/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('localIndexOwners: une entree planned ne possede rien', () => {
  // La regle vient du verificateur de nexus : compter les globs d'une entree planned
  // ferait baisser le plafond d'orphelins sans qu'un diagramme soit ecrit.
  const planned = [{ repo: 'nexus', entries: parseIndex('diagrams:\n  - name: nexus-z\n    status: planned\n    covers:\n      - "nexus:claude-code-api/src/main.rs"\n') }];
  assert.equal(localIndexOwners(planned, nexusFiles).size, 0);
  const verified = [{ repo: 'nexus', entries: parseIndex('diagrams:\n  - name: nexus-z\n    status: verified\n    covers:\n      - "nexus:claude-code-api/src/main.rs"\n') }];
  assert.equal(localIndexOwners(verified, nexusFiles).size, 1);
});

test('renderOrphans: le registre distingue orphelins et perimetres reserves', () => {
  const md = renderOrphans(['backend:src/a.rs', 'backend:src/b.rs'], 4, [], 1);
  assert.match(md, /\*\*2 orphelins sur 4 fichiers source \(50\.0 %\)\.\*\*/);
  assert.match(md, /SEULE une entree `verified` possede/);
  assert.match(md, /\*\*1 sont deja reserves\*\* par une entree `planned`/);
  assert.match(md, /acheterait du credit sur des intentions/);
});

test('gate: une entree planned ne retire PAS son perimetre des orphelins', () => {
  // Le coeur de l'honnetete du chiffre : l'index principal n'a que des entrees planned,
  // donc il ne doit rien couvrir. Sans cette regle, 550 fichiers disparaitraient du compte.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    const { code, out } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 0, out);
    // src/own.rs est couvert par po-x, mais po-x est `planned` : il reste orphelin.
    assert.match(out, /2 orphelins/);
    assert.match(out, /dont 1 reserves par une entree planned/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

// --- le cliquet d'orphelins
test('orphanCeiling: lit le marqueur, tolere les espaces, rejette le reste', () => {
  assert.equal(orphanCeiling('<!-- orphan-ceiling: 42 -->'), 42);
  assert.equal(orphanCeiling('texte <!--orphan-ceiling:7--> suite'), 7);
  assert.equal(orphanCeiling('<!-- orphan-ceiling: -->'), null);
  assert.equal(orphanCeiling('<!-- orphan-ceiling: beaucoup -->'), null);
  assert.equal(orphanCeiling('aucun marqueur'), null);
  assert.equal(orphanCeiling(null), null);
});

test('renderOrphans: publie le plafond et dit que le relever n est pas la sortie', () => {
  const md = renderOrphans(['backend:src/a.rs'], 4, [], 0, 7);
  assert.match(md, /<!-- orphan-ceiling: 7 -->/);
  assert.match(md, /Le plafond est \*\*7\*\*/);
  assert.match(md, /pas un plafond plus haut/);
});

function withCeiling(be, n) {
  const p = join(be, 'docs/diagrams/ORPHANS.md');
  writeFileSync(p, `<!-- orphan-ceiling: ${n} -->\n`);
}

test('cliquet: au-dessus du plafond, le gate ECHOUE avec la sortie indiquee', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    withCeiling(be, 1); // 2 orphelins reels
    const { code, out } = run(be, nx);
    assert.equal(code, 1);
    assert.match(out, /au-dessus du plafond de 1/);
    assert.match(out, /Relever le plafond n'est pas la sortie/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet: --write-orphans ne releve PAS le plafond tout seul', () => {
  // Le coeur du cliquet : s'il se releve a la regeneration, il ne tient rien.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    withCeiling(be, 1);
    const { code } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 1, 'la regeneration doit echouer, pas absoudre');
    assert.equal(orphanCeiling(readFileSync(join(be, 'docs/diagrams/ORPHANS.md'), 'utf8')), 1, 'plafond intact');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet: --raise-ceiling exige une raison, et la trace', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    withCeiling(be, 1);
    // sans raison (drapeau suivant), c'est refuse
    assert.equal(run(be, nx, ['--write-orphans', '--raise-ceiling', '--list']).code, 1);
    const { code, out } = run(be, nx, ['--write-orphans', '--raise-ceiling', 'raison ecrite']);
    assert.equal(code, 0, out);
    assert.match(out, /plafond d'orphelins releve 1 -> 2 : raison ecrite/);
    assert.equal(orphanCeiling(readFileSync(join(be, 'docs/diagrams/ORPHANS.md'), 'utf8')), 2);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet: il descend seul quand un fichier gagne un proprietaire', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: ['nexus:claude-code-api/src/core/model_registry.rs'] });
  try {
    withCeiling(be, 5); // large
    const { code } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 0);
    assert.equal(orphanCeiling(readFileSync(join(be, 'docs/diagrams/ORPHANS.md'), 'utf8')), 1, 'descendu a la valeur reelle');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet: supprimer le marqueur ne le desarme pas', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeFileSync(join(be, 'docs/diagrams/ORPHANS.md'), 'liste sans marqueur\n');
    const { code, out } = run(be, nx);
    assert.equal(code, 1);
    assert.match(out, /ne porte pas de marqueur/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

// --- portee incomplete : le cas de la CI d'un seul depot.
// Le plafond publie compte quatre depots ; un checkout partiel en voit moins. Comparer les
// deux chiffres serait un vert permanent qui ne verifie rien, et le regenerer detruirait le
// plafond. Ces deux tests sont la parce que la premiere version faisait exactement ca.
function partial(be, nx, args = []) {
  const r = spawnSync('node', [script, ...args], {
    encoding: 'utf8',
    env: { ...process.env, DIAGRAM_ROOT_BACKEND: be, DIAGRAM_ROOT_NEXUS: join(nx, 'absent'), DIAGRAM_ROOT_FRONTEND: join(nx, 'absent'), DIAGRAM_ROOT_WEBSITE: join(nx, 'absent') },
  });
  return { code: r.status, out: `${r.stdout ?? ''}${r.stderr ?? ''}` };
}

test('portee incomplete: le cliquet n est ni applique ni compare', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    withCeiling(be, 0); // un plafond que le compte partiel depasserait
    const { code, out } = partial(be, nx);
    assert.equal(code, 0, 'un index sain passe : le cliquet est hors sujet, pas viole');
    assert.match(out, /portee incomplete/);
    assert.match(out, /ne sont pas comparables au plafond/);
    assert.doesNotMatch(out, /au-dessus du plafond/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('portee incomplete: --write-orphans est REFUSE, le plafond survit', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    withCeiling(be, 1201); // plafond inter-depots
    const { code, out } = partial(be, nx, ['--write-orphans']);
    assert.equal(code, 1);
    assert.match(out, /refuse en portee incomplete/);
    assert.equal(orphanCeiling(readFileSync(join(be, 'docs/diagrams/ORPHANS.md'), 'utf8')), 1201, 'plafond intact');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('portee incomplete: un index casse echoue quand meme', () => {
  // La portee partielle ne doit pas devenir une amnistie generale.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/absent.rs'], localCovers: null });
  try {
    withCeiling(be, 1201);
    const { code, out } = partial(be, nx);
    assert.equal(code, 1);
    assert.match(out, /ne matche aucun fichier/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
