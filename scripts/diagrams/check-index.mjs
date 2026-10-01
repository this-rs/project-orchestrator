#!/usr/bin/env node
// Verifie docs/diagrams/INDEX.yml contre le code reel, hors reseau :
//   1. chaque glob `covers` matche au moins un fichier existant ;
//   2. un fichier source a AU PLUS UN diagramme proprietaire (pas de covers qui se recouvrent) ;
//   3. status verified => docs/diagrams/<name>.mmd existe, avec les en-tetes
//      `%% name`, `%% covers` (identique a l'index) et `%% verified` (sha court) ;
//      status planned => pas de fichier ; tout .mmd du dossier est dans l'index ;
//   4. l'index porte la carte d'index `po-carte` (role: index) et toute carte provisoire
//      reprise est declaree par `supersedes:` sur l'entree qui la remplace ;
//   5. la regle du proprietaire unique vaut ENTRE les index : un depot voisin peut tenir son
//      propre docs/diagrams/INDEX.yml pour les diagrammes dont le .mmd vit chez lui ; toute
//      collision entre cet index local et celui-ci est une erreur, et les fichiers qu'il
//      possede ne sont pas comptes comme orphelins ici ;
//   6. liste les fichiers source sans diagramme proprietaire (orphelins) et tient a jour
//      le registre publie docs/diagrams/ORPHANS.md.
//
// Usage : node scripts/diagrams/check-index.mjs [options]
//   --list                liste les orphelins fichier par fichier sur la sortie standard
//   --write-orphans       (re)genere docs/diagrams/ORPHANS.md
//   --check-orphans       echoue si docs/diagrams/ORPHANS.md n'est pas a jour (defaut : avertissement)
//   --fail-on-orphans     echoue s'il reste un orphelin (cible finale, pas l'etat actuel)
//   --strict              echoue aussi si un depot voisin est absent
//   --json <fichier>      ecrit un resume machine
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
const ORPHANS_DOC = 'docs/diagrams/ORPHANS.md';

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

// --- regle du proprietaire unique : un fichier couvert par deux diagrammes n'a pas de proprietaire.
// `owners` : Map<'repo:chemin', string[] (noms de diagrammes, dans l'ordre de l'index)>.
export function sharedOwnership(owners) {
  const out = [];
  for (const [file, names] of owners) {
    const uniq = [...new Set(names)];
    if (uniq.length > 1) out.push({ file, names: uniq });
  }
  return out.sort((a, b) => (a.file < b.file ? -1 : 1));
}

// --- registre publie des orphelins : texte deterministe, sans sha ni date (verifiable hors reseau).
export function renderOrphans(orphans, totalSources, localIndexes = []) {
  const byRepo = {};
  for (const o of orphans) { const [repo, path] = [o.slice(0, o.indexOf(':')), o.slice(o.indexOf(':') + 1)]; (byRepo[repo] ??= []).push(path); }
  const pct = totalSources ? ((orphans.length / totalSources) * 100).toFixed(1) : '0.0';
  const out = [
    '<!-- Genere par scripts/diagrams/check-index.mjs --write-orphans. Ne pas editer a la main. -->',
    '',
    '# Fichiers source sans diagramme proprietaire',
    '',
    'Un fichier source est **orphelin** quand aucun `covers` de `INDEX.yml` ne le matche :',
    'aucun diagramme ne repond de son comportement. Cette liste est publiee pour etre honnete',
    'sur ce que la cartographie couvre reellement — on ne reduit pas le denominateur, on la reduit elle.',
    '',
    `**${orphans.length} orphelins sur ${totalSources} fichiers source (${pct} %).**`,
    '',
    'Regeneration (hors reseau) :',
    '',
    '```',
    'node scripts/diagrams/check-index.mjs --write-orphans',
    '```',
    '',
  ];
  if (localIndexes.length) {
    out.push(
      'Un depot voisin peut tenir son propre index pour les diagrammes dont le `.mmd` vit chez lui.',
      'Les fichiers qu\'il possede ont un proprietaire et ne figurent donc pas ci-dessous ; toute',
      'collision entre les deux index est une erreur, pas un arrangement.',
      '',
    );
    for (const { repo, owned, diagrams } of [...localIndexes].sort((a, b) => (a.repo < b.repo ? -1 : 1))) {
      out.push(`- \`${repo}\` : ${diagrams} diagrammes, ${owned} fichiers possedes la-bas`);
    }
    out.push('');
  }
  for (const repo of Object.keys(byRepo).sort()) {
    const paths = byRepo[repo].sort();
    out.push(`## ${repo} (${paths.length})`, '');
    const byDir = {};
    for (const p of paths) { const d = p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '.'; (byDir[d] ??= []).push(p.slice(d === '.' ? 0 : d.length + 1)); }
    for (const d of Object.keys(byDir).sort()) out.push(`- \`${d}/\` (${byDir[d].length}) : ${byDir[d].sort().map((f) => `\`${f}\``).join(', ')}`);
    out.push('');
  }
  return out.join('\n');
}

// --- index locaux des depots voisins.
// Un depot peut tenir son propre `docs/diagrams/INDEX.yml` pour les diagrammes dont le `.mmd`
// vit chez lui (son gate tourne alors dans sa chaine d'outils, pas la notre). La regle du
// proprietaire unique vaut ENTRE les index : on resout leurs `covers` pour pouvoir detecter
// une collision avec le notre, et pour ne pas compter leurs fichiers comme orphelins.
// `files` : {repo: [chemins]}. Retourne Map<'repo:chemin', 'repo/INDEX.yml#nom'>.
export function localIndexOwners(indexes, files) {
  const owned = new Map();
  for (const { repo, entries } of indexes) {
    for (const e of entries) {
      for (const g of e.covers) {
        const m = g.match(/^([a-z]+):(.+)$/);
        // Un index local ne possede que des chemins de SON depot : un glob qui en designe un
        // autre serait une prise de pouvoir sur un depot voisin, on l'ignore.
        if (!m || m[1] !== repo) continue;
        const re = globToRegExp(m[2]);
        for (const f of (files[repo] ?? [])) if (re.test(f)) owned.set(`${repo}:${f}`, `${repo}/INDEX.yml#${e.name}`);
      }
    }
  }
  return owned;
}

function readLocalIndexes(files) {
  const out = [];
  for (const [repo, root] of Object.entries(roots)) {
    if (repo === 'backend' || !files[repo]) continue;
    const p = join(root, 'docs/diagrams/INDEX.yml');
    if (!existsSync(p)) continue;
    try {
      out.push({ repo, entries: parseIndex(readFileSync(p, 'utf8')) });
    } catch (err) {
      console.warn(`AVERTISSEMENT index local de '${repo}' illisible (${err.message}) : ses fichiers resteront orphelins ici`);
    }
  }
  return out;
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

  // Index locaux des depots voisins : leurs fichiers ont un proprietaire, ailleurs.
  const localIndexes = readLocalIndexes(files);
  const elsewhere = localIndexOwners(localIndexes, files);

  const owners = new Map();
  const problems = [];
  const names = new Set();
  const superseded = [];
  for (const e of entries) {
    if (names.has(e.name)) problems.push(`${e.name} : doublon dans l'index`);
    names.add(e.name);
    if (!e.owner) problems.push(`${e.name} : champ 'owner' manquant`);
    if (!['planned', 'verified'].includes(e.status)) problems.push(`${e.name} : status doit etre planned ou verified`);
    if (!/^(po-[a-z0-9-]+|[a-z0-9]+(-[a-z0-9]+)+)$/.test(e.name)) problems.push(`${e.name} : nom hors nomenclature`);
    if (e.supersedes) for (const old of e.supersedes.split(/[,\s]+/).filter(Boolean)) superseded.push({ old, by: e.name });
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
      for (const h of hits) { const k = `${repo}:${h}`; owners.set(k, [...(owners.get(k) ?? []), e.name]); }
    }
  }

  // La carte d'index : elle repond de INDEX.yml, et c'est a elle que se rattachent les cartes reprises.
  const indexCards = entries.filter((e) => e.role === 'index');
  if (indexCards.length !== 1) problems.push(`l'index doit porter exactement une entree 'role: index' (trouve : ${indexCards.length})`);
  else if (!indexCards[0].covers.some((g) => g.endsWith('docs/diagrams/INDEX.yml'))) problems.push(`${indexCards[0].name} : la carte d'index doit couvrir docs/diagrams/INDEX.yml`);
  for (const { old, by } of superseded) {
    if (names.has(old)) problems.push(`${by} : 'supersedes: ${old}' mais ${old} est encore une entree de l'index`);
    if (!/-\d+$/.test(old)) problems.push(`${by} : 'supersedes: ${old}' ne ressemble pas a une carte provisoire (suffixe numerique attendu)`);
  }

  // Un fichier couvert par deux diagrammes n'a pas de proprietaire : c'est une erreur, pas un detail.
  for (const { file, names: n } of sharedOwnership(owners)) problems.push(`${file} : couvert par ${n.length} diagrammes (${n.join(', ')}) ; un fichier n'a qu'un proprietaire`);

  // La meme regle vaut ENTRE les index : si un depot voisin revendique deja un fichier, cet
  // index ne doit pas le revendiquer aussi, sinon deux gates se contredisent sur le meme fichier.
  for (const [file, where] of elsewhere) {
    const here = owners.get(file);
    if (here) problems.push(`${file} : revendique par ${[...new Set(here)].join(', ')} ET par ${where} ; les deux index se contredisent`);
  }

  const dir = join(backendRoot, 'docs/diagrams');
  if (existsSync(dir)) for (const f of readdirSync(dir)) if (f.endsWith('.mmd') && !names.has(f.slice(0, -4))) problems.push(`${f} : diagramme absent de l'index`);

  const orphans = [];
  for (const [repo, list] of Object.entries(files)) {
    for (const f of list) {
      if (!SOURCE[repo].some((r) => r.test(f)) || NOT_SOURCE.some((r) => r.test(f))) continue;
      const key = `${repo}:${f}`;
      if (!owners.has(key) && !elsewhere.has(key)) orphans.push(key);
    }
  }
  orphans.sort();

  const totalSources = Object.entries(files).reduce((n, [repo, list]) => n + list.filter((f) => SOURCE[repo].some((r) => r.test(f)) && !NOT_SOURCE.some((r) => r.test(f))).length, 0);
  const nVerified = entries.filter((e) => e.status === 'verified').length;
  console.log(`${entries.length} diagrammes indexes (${nVerified} verified, ${entries.length - nVerified} planned) ; ${totalSources} fichiers source ; ${orphans.length} orphelins (sans diagramme proprietaire)`);
  for (const { repo, entries: le } of localIndexes) {
    const n = [...elsewhere.keys()].filter((k) => k.startsWith(`${repo}:`)).length;
    console.log(`  index local '${repo}' : ${le.length} diagrammes, ${n} fichiers possedes ailleurs (non comptes ici)`);
  }
  const byDir = {};
  for (const o of orphans) { const d = o.replace(/\/[^/]*$/, ''); (byDir[d] ??= []).push(o.slice(d.length + 1)); }
  const verbose = args.includes('--list');
  for (const [d, fs] of Object.entries(byDir).sort()) console.log(`  ${d}/ (${fs.length})${verbose ? ' ' + fs.join(' ') : ''}`);

  // Registre publie : on le regenere ou on verifie qu'il est a jour. Deterministe, donc
  // utilisable comme gate hors reseau ; il ne depend que de l'index et de la liste des fichiers.
  const orphansPath = join(backendRoot, ORPHANS_DOC);
  const rendered = renderOrphans(orphans, totalSources, localIndexes.map(({ repo, entries: le }) => ({
    repo,
    diagrams: le.length,
    owned: [...elsewhere.keys()].filter((k) => k.startsWith(`${repo}:`)).length,
  })));
  if (args.includes('--write-orphans')) {
    writeFileSync(orphansPath, rendered);
    console.log(`${ORPHANS_DOC} regenere (${orphans.length} orphelins)`);
  } else if (missingRepos.length) {
    console.warn(`AVERTISSEMENT ${ORPHANS_DOC} non verifie : des depots manquent (${missingRepos.join(', ')})`);
  } else {
    const current = existsSync(orphansPath) ? readFileSync(orphansPath, 'utf8') : null;
    if (current !== rendered) {
      const msg = `${ORPHANS_DOC} n'est pas a jour : relancer avec --write-orphans`;
      if (args.includes('--check-orphans')) problems.push(msg); else console.warn(`AVERTISSEMENT ${msg}`);
    }
  }

  if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ diagrams: entries.length, sources: totalSources, orphans }, null, 2));
  for (const p of problems) console.error(`ERREUR ${p}`);
  if (strict && missingRepos.length) { console.error(`ERREUR depots absents en mode --strict : ${missingRepos.join(', ')}`); process.exit(1); }
  if (problems.length) process.exit(1);
  if (args.includes('--fail-on-orphans') && orphans.length) process.exit(1);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
