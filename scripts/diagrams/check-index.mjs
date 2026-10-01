#!/usr/bin/env node
// Verifie docs/diagrams/INDEX.yml contre le code reel, hors reseau :
//   1. chaque glob `covers` matche au moins un fichier existant ;
//   2. status verified => docs/diagrams/<name>.mmd existe, avec les en-tetes
//      `%% name`, `%% covers` (identique a l'index) et `%% verified` (sha court) ;
//      status planned => pas de fichier ; tout .mmd du dossier est dans l'index ;
//   3. liste les fichiers source sans diagramme proprietaire (orphelins).
//
// Usage : node scripts/diagrams/check-index.mjs [--strict] [--list] [--json <fichier>] [--fail-on-orphans]
//
// Les globs `covers` sont prefixes par le depot : backend: frontend: nexus: website:
// Racines (variables d'environnement, sinon valeurs par defaut) :
//   DIAGRAM_ROOT_BACKEND  (defaut : la racine de ce depot)
//   DIAGRAM_ROOT_FRONTEND (defaut : ../frontend)   DIAGRAM_ROOT_NEXUS (../nexus)   DIAGRAM_ROOT_WEBSITE (../website)
// Un depot absent est ignore avec un avertissement, sauf avec --strict (exit 1).
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const backendRoot = resolve(process.env.DIAGRAM_ROOT_BACKEND ?? join(here, '..', '..'));
const roots = {
  backend: backendRoot,
  frontend: resolve(process.env.DIAGRAM_ROOT_FRONTEND ?? join(backendRoot, '..', 'frontend')),
  nexus: resolve(process.env.DIAGRAM_ROOT_NEXUS ?? join(backendRoot, '..', 'nexus')),
  website: resolve(process.env.DIAGRAM_ROOT_WEBSITE ?? join(backendRoot, '..', 'website')),
};

// Fichiers "source" soumis a la regle du proprietaire (par depot).
const SOURCE = {
  backend: [/^src\/.*\.rs$/, /^crates\/[^/]+\/src\/.*\.rs$/, /^desktop\/src-tauri\/src\/.*\.rs$/],
  frontend: [/^src\/.*\.tsx?$/],
  nexus: [/^claude-code-sdk-rs\/src\/.*\.rs$/, /^claude-code-api\/src\/.*\.rs$/],
  website: [/^src\/.*\.tsx?$/],
};
const NOT_SOURCE = [/(^|\/)__tests__\//, /\.test\.tsx?$/, /\.d\.ts$/, /(^|\/)test_helpers\.rs$/, /_tests?\.rs$/, /(^|\/)tests?\.rs$/];

// --- glob -> RegExp : ** (profondeur), * (un segment), ? , {a,b}
export function globToRegExp(glob) {
  let re = '';
  let depth = 0;
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (c === '*') {
      if (glob[i + 1] === '*') {
        i++;
        if (glob[i + 1] === '/') { i++; re += '(?:.*/)?'; } else re += '.*';
      } else re += '[^/]*';
    } else if (c === '?') re += '[^/]';
    else if (c === '{') { depth++; re += '(?:'; }
    else if (c === '}' && depth > 0) { depth--; re += ')'; }
    else if (c === ',' && depth > 0) re += '|';
    else re += c.replace(/[.+^$()|[\]\\]/g, '\\$&');
  }
  return new RegExp(`^${re}$`);
}

// --- en-tete d'un .mmd : trois premieres lignes `%% name|covers|verified: ...`
export function parseHeader(text) {
  const h = {};
  for (const line of text.split('\n').slice(0, 3)) {
    const m = line.match(/^%% (name|covers|verified):\s*(.*?)\s*$/);
    if (m) h[m[1]] = m[2];
  }
  return h;
}

export function checkDiagramFile(entry, text) {
  const errs = [];
  const h = parseHeader(text);
  if (h.name !== entry.name) errs.push(`${entry.name} : en-tete '%% name' absent ou different (${h.name ?? 'absent'})`);
  const want = [...entry.covers].sort().join(' ');
  const got = (h.covers ?? '').split(/\s+/).filter(Boolean).sort().join(' ');
  if (got !== want) errs.push(`${entry.name} : '%% covers' differe de l'index`);
  if (!/^[0-9a-f]{7,40}$/.test(h.verified ?? '')) errs.push(`${entry.name} : '%% verified' doit etre un sha git court`);
  return errs;
}

// --- lecture de INDEX.yml (sous-ensemble strict, sans dependance)
export function parseIndex(text) {
  const entries = [];
  let cur = null;
  let list = null;
  for (const raw of text.split('\n')) {
    if (!raw.trim() || raw.trim().startsWith('#') || raw.trim() === 'diagrams:') continue;
    let m;
    if ((m = raw.match(/^ {2}- name:\s*(.+?)\s*$/))) { cur = { name: m[1], covers: [] }; entries.push(cur); list = null; continue; }
    if (!cur) throw new Error(`INDEX.yml : ligne hors entree : ${raw}`);
    if ((m = raw.match(/^ {4}covers:\s*$/))) { list = cur.covers; continue; }
    if ((m = raw.match(/^ {6}- "?(.+?)"?\s*$/)) && list) { list.push(m[1]); continue; }
    if ((m = raw.match(/^ {4}([a-z_]+):\s*(.*?)\s*$/))) { list = null; cur[m[1]] = m[2]; continue; }
    throw new Error(`INDEX.yml : ligne non reconnue : ${raw}`);
  }
  return entries;
}

function listFiles(root) {
  if (existsSync(join(root, '.git'))) {
    return execFileSync('git', ['-C', root, 'ls-files', '--cached', '--others', '--exclude-standard'], { encoding: 'utf8', maxBuffer: 1 << 28 }).split('\n').filter(Boolean)
      .filter((f) => existsSync(join(root, f)));
  }
  const out = [];
  const walk = (d) => {
    for (const e of readdirSync(d)) {
      if (e === 'node_modules' || e === 'dist' || e === '.git') continue;
      const p = join(d, e);
      statSync(p).isDirectory() ? walk(p) : out.push(relative(root, p));
    }
  };
  walk(root);
  return out;
}

function main() {
  const args = process.argv.slice(2);
  const strict = args.includes('--strict');
  const jsonOut = args.includes('--json') ? args[args.indexOf('--json') + 1] : null;
  const entries = parseIndex(readFileSync(join(backendRoot, 'docs/diagrams/INDEX.yml'), 'utf8'));

  const files = {};
  const missingRepos = [];
  for (const [repo, root] of Object.entries(roots)) {
    if (!existsSync(root)) { missingRepos.push(repo); continue; }
    files[repo] = listFiles(root);
  }
  for (const r of missingRepos) console.warn(`AVERTISSEMENT depot '${r}' introuvable (${roots[r]}) : globs correspondants non verifies`);

  const owned = new Set();
  const problems = [];
  const names = new Set();
  for (const e of entries) {
    if (names.has(e.name)) problems.push(`${e.name} : doublon dans l'index`);
    names.add(e.name);
    if (!e.owner) problems.push(`${e.name} : champ 'owner' manquant`);
    if (!['planned', 'verified'].includes(e.status)) problems.push(`${e.name} : status doit etre planned ou verified`);
    if (!/^(po-[a-z0-9-]+|[a-z0-9]+(-[a-z0-9]+)+)$/.test(e.name)) problems.push(`${e.name} : nom hors nomenclature`);
    const file = join(backendRoot, 'docs/diagrams', `${e.name}.mmd`);
    if (e.status === 'verified') {
      if (!existsSync(file)) problems.push(`${e.name} : status verified mais docs/diagrams/${e.name}.mmd est absent`);
      else problems.push(...checkDiagramFile(e, readFileSync(file, 'utf8')));
    } else if (existsSync(file)) problems.push(`${e.name} : fichier present, status devrait etre verified`);
    if (!e.covers.length) problems.push(`${e.name} : covers vide`);
    for (const g of e.covers) {
      const m = g.match(/^(backend|frontend|nexus|website):(.+)$/);
      if (!m) { problems.push(`${e.name} : glob sans prefixe de depot : ${g}`); continue; }
      const [, repo, pat] = m;
      if (!files[repo]) continue;
      const re = globToRegExp(pat);
      const hits = files[repo].filter((f) => re.test(f));
      if (!hits.length) problems.push(`${e.name} : le glob ne matche aucun fichier : ${g}`);
      for (const h of hits) owned.add(`${repo}:${h}`);
    }
  }

  const dir = join(backendRoot, 'docs/diagrams');
  if (existsSync(dir)) for (const f of readdirSync(dir)) if (f.endsWith('.mmd') && !names.has(f.slice(0, -4))) problems.push(`${f} : diagramme absent de l'index`);

  const orphans = [];
  for (const [repo, list] of Object.entries(files)) {
    for (const f of list) {
      if (!SOURCE[repo].some((r) => r.test(f)) || NOT_SOURCE.some((r) => r.test(f))) continue;
      if (!owned.has(`${repo}:${f}`)) orphans.push(`${repo}:${f}`);
    }
  }
  orphans.sort();

  const totalSources = Object.entries(files).reduce((n, [repo, list]) => n + list.filter((f) => SOURCE[repo].some((r) => r.test(f)) && !NOT_SOURCE.some((r) => r.test(f))).length, 0);
  const nVerified = entries.filter((e) => e.status === 'verified').length;
  console.log(`${entries.length} diagrammes indexes (${nVerified} verified, ${entries.length - nVerified} planned) ; ${totalSources} fichiers source ; ${orphans.length} orphelins (sans diagramme proprietaire)`);
  const byDir = {};
  for (const o of orphans) { const d = o.replace(/\/[^/]*$/, ''); (byDir[d] ??= []).push(o.slice(d.length + 1)); }
  const verbose = args.includes('--list');
  for (const [d, fs] of Object.entries(byDir).sort()) console.log(`  ${d}/ (${fs.length})${verbose ? ' ' + fs.join(' ') : ''}`);
  if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ diagrams: entries.length, sources: totalSources, orphans }, null, 2));
  for (const p of problems) console.error(`ERREUR ${p}`);
  if (strict && missingRepos.length) { console.error(`ERREUR depots absents en mode --strict : ${missingRepos.join(', ')}`); process.exit(1); }
  if (problems.length) process.exit(1);
  if (args.includes('--fail-on-orphans') && orphans.length) process.exit(1);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
