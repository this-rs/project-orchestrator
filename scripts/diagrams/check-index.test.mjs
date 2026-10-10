import { test } from 'node:test';
import assert from 'node:assert/strict';
import { globToRegExp, parseIndex, parseHeader, checkDiagramFile, sharedOwnership, renderOrphans, localIndexOwners, orphanCeiling, orphanCeilingsByRepo, orphansByRepo, isExternal, checkExternalEntry, looksLikeHost, commitStatus } from './check-index.mjs';

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
import { spawnSync, execFileSync } from 'node:child_process';
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
    assert.match(out, /le plafond TOTAL \(0\) n'est ni applique ni mis a jour/);
    // sans marqueur par depot, rien n'est applique non plus : le total seul ne suffit pas
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

// --- cliquet PAR DEPOT : c'est lui qui rend le mecanisme reel en CI.
test('orphanCeilingsByRepo / orphansByRepo', () => {
  assert.deepEqual(orphanCeilingsByRepo('<!-- orphan-ceiling-backend: 456 -->\n<!-- orphan-ceiling-nexus: 73 -->'), { backend: 456, nexus: 73 });
  assert.deepEqual(orphanCeilingsByRepo('<!-- orphan-ceiling: 1201 -->'), {}, 'le total n est pas un depot');
  assert.deepEqual(orphanCeilingsByRepo(''), {});
  assert.deepEqual(orphansByRepo(['backend:a.rs', 'backend:b.rs', 'nexus:c.rs']), { backend: 2, nexus: 1 });
  assert.deepEqual(orphansByRepo([]), {});
});

test('renderOrphans: ecrit un marqueur par depot, trie', () => {
  const md = renderOrphans(['nexus:src/b.rs', 'backend:src/a.rs', 'backend:src/c.rs'], 9, [], 0, 3);
  assert.match(md, /<!-- orphan-ceiling: 3 -->/);
  assert.match(md, /<!-- orphan-ceiling-backend: 2 -->/);
  assert.match(md, /<!-- orphan-ceiling-nexus: 1 -->/);
  assert.ok(md.indexOf('ceiling-backend') < md.indexOf('ceiling-nexus'), 'ordre stable');
});

test('cliquet par depot: applique MEME en portee incomplete', () => {
  // Le point de tout l'exercice : en CI un seul depot est la, et le cliquet doit quand meme
  // mordre. Avant les plafonds par depot, il ne mordait jamais dans cette situation.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeFileSync(join(be, 'docs/diagrams/ORPHANS.md'), '<!-- orphan-ceiling: 1201 -->\n<!-- orphan-ceiling-backend: 0 -->\n');
    const { code, out } = partial(be, nx);
    assert.equal(code, 1, 'un orphelin backend au-dessus de 0 doit echouer');
    assert.match(out, /backend : 1 fichiers source sans diagramme proprietaire, au-dessus du plafond de 0/);
    assert.match(out, /le plafond TOTAL .* n'est ni applique/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet par depot: sous le plafond, il passe en portee incomplete', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeFileSync(join(be, 'docs/diagrams/ORPHANS.md'), '<!-- orphan-ceiling: 1201 -->\n<!-- orphan-ceiling-backend: 5 -->\n');
    const { code, out } = partial(be, nx);
    assert.equal(code, 0, out);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet par depot: un depot absent n est pas tenu a son plafond', () => {
  // nexus absent du checkout : son plafond ne doit pas faire echouer la CI du backend.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeFileSync(join(be, 'docs/diagrams/ORPHANS.md'), '<!-- orphan-ceiling: 1201 -->\n<!-- orphan-ceiling-backend: 5 -->\n<!-- orphan-ceiling-nexus: 0 -->\n');
    const { code, out } = partial(be, nx);
    assert.equal(code, 0, out);
    assert.doesNotMatch(out, /nexus : /);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

// --- Index voisin DERIVE (format de nexus : scripts/derive_diagram_index.py) ---------------
// Trois differences avec cet index : une section `orphans:` apres `diagrams:`, des globs sans
// prefixe de depot, et des entrees sans `status` (l'entree existe parce que le .mmd existe,
// `file` + `verified` a la place). Mesure du 02/10/2026 : l'index de nexus etait declare
// « illisible », ses 77 fichiers possedes comptes orphelins ici, et son plafond releve.

const derivedIndex = `# genere\ndiagrams:\n  - name: nexus-sdk-types\n    file: nexus-sdk-types.mmd\n    verified: 2026-10-01\n    covers:\n      - claude-code-api/src/core/*.rs\n\norphans:\n  - claude-code-api/src/bin/ccapi.rs\n`;

test('parseIndex: un index derive avec une section orphans: est lu, et orphans: ignoree', () => {
  const entries = parseIndex(derivedIndex);
  assert.equal(entries.length, 1);
  assert.equal(entries[0].name, 'nexus-sdk-types');
  assert.equal(entries[0].file, 'nexus-sdk-types.mmd');
  assert.deepEqual(entries[0].covers, ['claude-code-api/src/core/*.rs']);
});

test('localIndexOwners: une entree derivee (file, sans status, glob sans prefixe) possede ses fichiers', () => {
  const owned = localIndexOwners([{ repo: 'nexus', entries: parseIndex(derivedIndex) }], nexusFiles);
  assert.equal(owned.get('nexus:claude-code-api/src/core/model_registry.rs'), 'nexus/INDEX.yml#nexus-sdk-types');
});

test('localIndexOwners: un glob sans prefixe ne possede que son propre depot', () => {
  const idx = [{ repo: 'nexus', entries: [{ name: 'x', file: 'x.mmd', covers: ['src/**'] }] }];
  assert.equal(localIndexOwners(idx, { nexus: ['claude-code-api/src/a.rs'], backend: ['src/a.rs'] }).size, 0);
});

test('gate: les fichiers possedes par un index voisin DERIVE ne sont pas des orphelins ici', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    mkdirSync(join(nx, 'docs/diagrams'), { recursive: true });
    writeFileSync(join(nx, 'docs/diagrams/INDEX.yml'), derivedIndex);
    const { code, out } = run(be, nx, ['--list']);
    assert.equal(code, 0, out);
    assert.doesNotMatch(out, /illisible/);
    assert.doesNotMatch(out, /nexus:claude-code-api\/src\/core\/model_registry\.rs/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet par depot: --write-orphans ne releve PAS un plafond de depot, et refuse d ecrire', () => {
  // Rejoue sans le correctif : le registre etait reecrit depuis le compte reel, donc
  // `orphan-ceiling-backend` passait de 0 a 1 par la commande meme censee l'en empecher.
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    const p = join(be, 'docs/diagrams/ORPHANS.md');
    writeFileSync(p, '<!-- orphan-ceiling: 1201 -->\n<!-- orphan-ceiling-backend: 0 -->\n');
    const { code, out } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 1, out);
    assert.match(out, /backend : 1 fichiers source sans diagramme proprietaire, au-dessus du plafond de 0/);
    assert.match(out, /--write-orphans refuse/);
    assert.deepEqual(orphanCeilingsByRepo(readFileSync(p, 'utf8')), { backend: 0 }, 'plafond de depot intact, registre non reecrit');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('cliquet par depot: --raise-ceiling avec raison releve aussi le plafond de depot, et le trace', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    const p = join(be, 'docs/diagrams/ORPHANS.md');
    writeFileSync(p, '<!-- orphan-ceiling: 1201 -->\n<!-- orphan-ceiling-backend: 0 -->\n');
    const { code, out } = run(be, nx, ['--write-orphans', '--raise-ceiling', 'trois fichiers ajoutes en amont']);
    assert.equal(code, 0, out);
    assert.match(out, /plafond de backend releve 0 -> 1 : trois fichiers ajoutes en amont/);
    assert.equal(orphanCeilingsByRepo(readFileSync(p, 'utf8')).backend, 1);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

// --- diagrammes EXTERNES : ils vivent dans le service Mermaid de l'equipe, le depot ne garde que
// l'entree. Depot public : aucun hote dans l'entree, seulement l'identifiant et l'emplacement.
const ext = (extra) => parseIndex(`diagrams:\n  - name: po-x\n    owner: t\n${extra}    covers:\n      - "backend:src/own.rs"\n`)[0];
const EXT_OK = '    status: verified\n    mermaid_id: abc123\n    external: po/doc/po-x@2\n    verified_at: 2026-10-10\n    verified_sha: 6c79054e\n';

test('externe: une entree verified complete ne leve rien', () => {
  const e = ext(EXT_OK);
  assert.ok(isExternal(e));
  assert.deepEqual(checkExternalEntry(e), []);
  // mermaid_id seul suffit, external seul aussi
  assert.deepEqual(checkExternalEntry(ext(EXT_OK.replace(/ {4}external:.*\n/, ''))), []);
  assert.deepEqual(checkExternalEntry(ext(EXT_OK.replace(/ {4}mermaid_id:.*\n/, ''))), []);
  assert.ok(!isExternal(ext('    status: verified\n')));
});

test('externe: identifiant absent, adresse hors site, releve incomplet ou planned sont des erreurs', () => {
  assert.match(checkExternalEntry(ext(EXT_OK.replace('mermaid_id: abc123', 'mermaid_id:').replace(/ {4}external:.*\n/, ''))).join('\n'), /'mermaid_id' vide/);
  assert.match(checkExternalEntry(ext(EXT_OK.replace('mermaid_id: abc123', 'mermaid_id:').replace(/ {4}external:.*\n/, ''))).join('\n'), /sans identifiant/);
  // une adresse avec hote est refusee : le depot est public, l'hote ne s'y ecrit pas
  assert.match(checkExternalEntry(ext(EXT_OK.replace('po/doc/po-x@2', 'https://example.com/po/doc/po-x@2'))).join('\n'), /sans hote/);
  assert.match(checkExternalEntry(ext(EXT_OK.replace('po/doc/po-x@2', 'po/doc/po-x'))).join('\n'), /sans hote/);
  // l'emplacement doit designer CE diagramme
  assert.match(checkExternalEntry(ext(EXT_OK.replace('po/doc/po-x@2', 'po/doc/po-y@2'))).join('\n'), /designe le diagramme po-y/);
  assert.match(checkExternalEntry(ext(EXT_OK.replace('6c79054e', 'HEAD'))).join('\n'), /verified_sha/);
  assert.match(checkExternalEntry(ext(EXT_OK.replace(/ {4}verified_at:.*\n/, ''))).join('\n'), /verified_at/);
  assert.match(checkExternalEntry(ext(EXT_OK.replace('status: verified', 'status: planned'))).join('\n'), /status devrait etre verified/);
});

test('externe: un nom d\'hote dans l\'emplacement est refuse, meme sans point', () => {
  for (const loc of ['localhost/doc/po-x@2', 'po/127.0.0.1/po-x@2', 'po/diagrams.example.com/po-x@2']) {
    assert.match(checkExternalEntry(ext(EXT_OK.replace('po/doc/po-x@2', loc))).join('\n'), /nom d'hote/, loc);
  }
  assert.ok(looksLikeHost('LocalHost'));
  for (const segment of ['po', 'architecture-po', 'doc.v2', 'project-orchestrator']) assert.ok(!looksLikeHost(segment), segment);
});

// Un depot git jetable avec un commit : `verified_sha` est verifie contre lui.
function gitRepoWithCommit(dir) {
  const git = (...args) => execFileSync('git', ['-C', dir, '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', ...args], { encoding: 'utf8' }).trim();
  git('init', '-q');
  git('add', '-A');
  git('commit', '-q', '-m', 'fixture');
  return git('rev-parse', 'HEAD');
}

// Un commit sur une branche laterale, puis retour sur la branche de depart : il existe, mais
// HEAD n'en descend pas (une branche non fusionnee, ou ecrasee par un squash).
function sideCommit(dir) {
  const git = (...args) => execFileSync('git', ['-C', dir, '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', ...args], { encoding: 'utf8' }).trim();
  const start = git('rev-parse', '--abbrev-ref', 'HEAD');
  git('switch', '-q', '-c', 'side');
  git('commit', '-q', '--allow-empty', '-m', 'side');
  const side = git('rev-parse', 'HEAD');
  git('switch', '-q', start);
  return side;
}

test('verified_sha: ancetre de HEAD, hors historique, absent, ou non verifiable hors depot', () => {
  const root = mkdtempSync(join(tmpdir(), 'diagram-sha-'));
  try {
    const repo = join(root, 'repo');
    mkdirSync(repo);
    writeFileSync(join(repo, 'f'), 'x\n');
    const head = gitRepoWithCommit(repo);
    assert.equal(commitStatus(repo, head.slice(0, 8)), 'ancestor');
    assert.equal(commitStatus(repo, sideCommit(repo).slice(0, 8)), 'not-ancestor');
    assert.equal(commitStatus(repo, 'deadbeef'), 'absent');
    const plain = join(root, 'plain');
    mkdirSync(plain);
    assert.equal(commitStatus(plain, head.slice(0, 8)), 'unknown');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: un verified_sha qui ne designe aucun commit du depot echoue', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeIndex(be, EXT_OK);
    const head = gitRepoWithCommit(be);
    const absent = run(be, nx);
    assert.equal(absent.code, 1, absent.out);
    assert.match(absent.out, /po-x : 'verified_sha: 6c79054e' ne designe aucun commit de ce depot/);
    writeIndex(be, EXT_OK.replace('6c79054e', head.slice(0, 8)));
    const present = run(be, nx);
    assert.doesNotMatch(present.out, /ne designe aucun commit/, present.out);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: un verified_sha hors de l\'historique de HEAD echoue', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeIndex(be, EXT_OK);
    const head = gitRepoWithCommit(be);
    const side = sideCommit(be);
    writeIndex(be, EXT_OK.replace('6c79054e', side.slice(0, 8)));
    const off = run(be, nx);
    assert.equal(off.code, 1, off.out);
    assert.match(off.out, /po-x : 'verified_sha: [0-9a-f]{8}' est hors de l'historique de HEAD/);
    writeIndex(be, EXT_OK.replace('6c79054e', head.slice(0, 8)));
    const on = run(be, nx);
    assert.doesNotMatch(on.out, /hors de l'historique|ne designe aucun commit/, on.out);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

function writeIndex(be, poX) {
  writeFileSync(join(be, 'docs/diagrams/INDEX.yml'),
    `diagrams:\n  - name: po-carte\n    owner: t\n    status: planned\n    role: index\n    covers:\n      - "backend:docs/diagrams/INDEX.yml"\n  - name: po-x\n    owner: t\n${poX}    covers:\n      - "backend:src/own.rs"\n`);
}

test('gate: une entree EXTERNE verified possede ses fichiers sans .mmd local', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeIndex(be, EXT_OK);
    const { code, out } = run(be, nx, ['--write-orphans']);
    assert.equal(code, 0, out);
    // src/own.rs a un proprietaire : il ne reste que le fichier de nexus.
    assert.match(out, /1 orphelins/);
    assert.match(out, /1 verified/);
    assert.doesNotMatch(readFileSync(join(be, 'docs/diagrams/ORPHANS.md'), 'utf8'), /own\.rs/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: une entree externe sans identifiant echoue, et ne possede rien', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeIndex(be, EXT_OK.replace('mermaid_id: abc123', 'mermaid_id:').replace(/ {4}external:.*\n/, ''));
    const { code, out } = run(be, nx);
    assert.equal(code, 1, out);
    assert.match(out, /po-x : diagramme externe sans identifiant/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: une entree externe ET un .mmd local : deux sources, refuse', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeIndex(be, EXT_OK);
    writeFileSync(join(be, 'docs/diagrams/po-x.mmd'), '%% name: po-x\n%% covers: backend:src/own.rs\n%% verified: 6c79054e\nflowchart TD\n');
    const { code, out } = run(be, nx);
    assert.equal(code, 1, out);
    assert.match(out, /diagramme externe ET docs\/diagrams\/po-x\.mmd present/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('gate: le proprietaire unique vaut aussi pour une entree externe', () => {
  const { root, be, nx } = fixture({ mainCovers: ['backend:src/own.rs'], localCovers: null });
  try {
    writeFileSync(join(be, 'docs/diagrams/INDEX.yml'),
      `diagrams:\n  - name: po-carte\n    owner: t\n    status: planned\n    role: index\n    covers:\n      - "backend:docs/diagrams/INDEX.yml"\n  - name: po-x\n    owner: t\n${EXT_OK}    covers:\n      - "backend:src/own.rs"\n  - name: po-y\n    owner: t\n    status: planned\n    covers:\n      - "backend:src/*.rs"\n`);
    const { code, out } = run(be, nx);
    assert.equal(code, 1, out);
    assert.match(out, /backend:src\/own\.rs : couvert par 2 diagrammes \(po-x, po-y\)/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
