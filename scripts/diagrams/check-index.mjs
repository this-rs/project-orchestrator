#!/usr/bin/env node
// Verifie docs/diagrams/INDEX.yml contre le code reel, hors reseau :
//   1. chaque glob `covers` matche au moins un fichier existant ;
//   2. un fichier source a AU PLUS UN diagramme proprietaire (pas de covers qui se recouvrent) ;
//   3. status verified => docs/diagrams/<name>.mmd existe, avec les en-tetes
//      `%% name`, `%% covers` (identique a l'index) et `%% verified` (sha court) ;
//      status planned => pas de fichier ; tout .mmd du dossier est dans l'index ;
//      ou bien le diagramme est EXTERNE (service Mermaid de l'equipe, regle par defaut) : l'entree
//      porte `mermaid_id` et/ou `external` (emplacement sans hote), plus `verified_at` et `verified_sha`, et aucun .mmd local ;
//      `verified_sha` doit designer un commit qui existe dans ce depot (`git cat-file -e`, hors reseau) ;
//      une entree externe verified possede ses `covers` comme une entree locale (hors reseau) ;
//   4. l'index porte la carte d'index `po-carte` (role: index) et toute carte provisoire
//      reprise est declaree par `supersedes:` sur l'entree qui la remplace ;
//   5. la regle du proprietaire unique vaut ENTRE les index : un depot voisin peut tenir son
//      propre docs/diagrams/INDEX.yml pour les diagrammes dont le .mmd vit chez lui ; toute
//      collision entre cet index local et celui-ci est une erreur, et les fichiers qu'il
//      possede ne sont pas comptes comme orphelins ici ;
//   6. SEULE une entree `verified` possede : une entree `planned` ne designe aucun fichier
//      ecrit, donc elle ne retire rien du compte des orphelins. Compter ses globs ferait
//      baisser le plafond d'orphelins sans qu'un seul diagramme soit ecrit : l'index
//      acheterait du credit sur des intentions. Les `covers` d'une entree `planned` servent
//      uniquement a reserver un perimetre (regle du proprietaire unique, point 2) ;
//   7. liste les fichiers source sans diagramme proprietaire (orphelins) et tient a jour
//      le registre publie docs/diagrams/ORPHANS.md.
//
// Usage : node scripts/diagrams/check-index.mjs [options]
//   --list                liste les orphelins fichier par fichier sur la sortie standard
//   --write-orphans       (re)genere docs/diagrams/ORPHANS.md
//   --check-orphans       echoue si docs/diagrams/ORPHANS.md n'est pas a jour (defaut : avertissement)
//   --fail-on-orphans     echoue s'il reste un orphelin (cible finale, pas l'etat actuel)
//   --raise-ceiling "<raison>"  releve le plafond d'orphelins (a eviter : la sortie est un glob)
//   --strict              echoue aussi si un depot voisin est absent
//   --json <fichier>      ecrit un resume machine
//
// Les globs `covers` sont prefixes par le depot : backend: frontend: nexus: website:
// Racines (variables d'environnement, sinon valeurs par defaut) :
//   DIAGRAM_ROOT_BACKEND  (defaut : la racine de ce depot)
//   DIAGRAM_ROOT_FRONTEND (defaut : ../frontend)   DIAGRAM_ROOT_NEXUS (../nexus)   DIAGRAM_ROOT_WEBSITE (../website)
// Un depot absent est ignore avec un avertissement, sauf avec --strict (exit 1).
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';
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

// --- diagramme EXTERNE : il vit dans le service Mermaid prive de l'equipe, le depot ne garde
// que son entree d'index. Ce depot est PUBLIC : l'hote du service ne doit y figurer nulle part
// (scripts/forbidden/check-forbidden-tokens.mjs) ; il est nomme hors depot (skill mermaid-design).
// L'entree porte donc l'identifiant du diagramme (`mermaid_id`) et, au choix, son emplacement
// SANS hote (`external: <workspace>/<session>/<name>@<version>`). Une entree est externe des
// qu'elle porte l'une de ces cles, meme vide : une cle vide est une erreur, pas une entree locale.
// Le controle reste HORS RESEAU : on verifie la forme de l'entree, jamais le service.
// `verified_sha` est le sha du backend contre lequel le diagramme a ete relu : c'est ce qui
// remplace l'en-tete `%% verified` d'un .mmd local.
// Un segment d'emplacement qui est un nom d'hote : la forme <workspace>/<session>/... admet
// `localhost/doc/po-x@2` (le workspace n'a pas de point), et une session peut porter des
// points. On refuse donc les noms d'hote locaux SANS point (localhost et ses alias), une
// adresse IPv4, et un segment qui finit comme un domaine (`.xx` et plus, lettres seules).
// Ce que ce controle ne peut pas voir : un nom d'hote d'intranet sans point qui ne serait pas
// dans cette liste ressemble a un workspace ; le gate des jetons interdits reste la defense.
const LOCAL_HOSTNAMES = new Set(['localhost', 'ip6-localhost', 'ip6-loopback', 'broadcasthost']);
export function looksLikeHost(segment) {
  const s = segment.toLowerCase();
  return LOCAL_HOSTNAMES.has(s) || /^\d{1,3}(\.\d{1,3}){3}$/.test(s) || /\.[a-z]{2,}$/.test(s);
}

// `verified_sha` doit DESIGNER un commit de ce depot : un sha bien forme mais absent (une
// branche ecrasee par un squash, une faute de frappe) passerait le controle de forme et ne
// prouverait rien. Verifie HORS RESEAU (`git cat-file -e <sha>^{commit}`) :
//   'present' : le commit existe ; 'absent' : il n'existe pas ;
//   'unknown' : pas un depot git, ou un clone superficiel (le commit peut exister en amont).
export function commitStatus(root, sha) {
  const git = (args) => spawnSync('git', ['-C', root, ...args], { encoding: 'utf8' });
  const inside = git(['rev-parse', '--is-inside-work-tree']);
  if (inside.status !== 0 || inside.stdout.trim() !== 'true') return 'unknown';
  if (git(['cat-file', '-e', `${sha}^{commit}`]).status === 0) return 'present';
  return git(['rev-parse', '--is-shallow-repository']).stdout.trim() === 'true' ? 'unknown' : 'absent';
}

export function isExternal(entry) {
  return Object.hasOwn(entry, 'mermaid_id') || Object.hasOwn(entry, 'external');
}

export function checkExternalEntry(entry) {
  const errs = [];
  const id = entry.mermaid_id ?? '';
  const loc = entry.external ?? '';
  if (Object.hasOwn(entry, 'mermaid_id') && !/^[A-Za-z0-9_-]+$/.test(id)) {
    errs.push(`${entry.name} : 'mermaid_id' vide ou invalide : un diagramme externe doit nommer son identifiant dans le service`);
  }
  if (Object.hasOwn(entry, 'external')) {
    const m = /^([a-z0-9-]+)\/([A-Za-z0-9._-]+)\/([a-z0-9-]+)@(\d+)$/.exec(loc);
    if (!m) errs.push(`${entry.name} : 'external' doit etre <workspace>/<session>/<name>@<version>, sans hote (trouve : ${loc || 'vide'})`);
    else if (looksLikeHost(m[1]) || looksLikeHost(m[2])) errs.push(`${entry.name} : 'external' commence par un nom d'hote (${looksLikeHost(m[1]) ? m[1] : m[2]}) ; l'emplacement s'ecrit sans hote`);
    else if (m[3] !== entry.name) errs.push(`${entry.name} : 'external' designe le diagramme ${m[3]}, pas ${entry.name}`);
  }
  if (!id && !loc) errs.push(`${entry.name} : diagramme externe sans identifiant ('mermaid_id' ou 'external')`);
  if (entry.status === 'verified') {
    if (!/^[0-9a-f]{7,40}$/.test(entry.verified_sha ?? '')) errs.push(`${entry.name} : 'verified_sha' doit etre le sha git (court) du code contre lequel le diagramme a ete relu`);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(entry.verified_at ?? '')) errs.push(`${entry.name} : 'verified_at' doit etre une date AAAA-MM-JJ`);
  } else {
    errs.push(`${entry.name} : diagramme externe publie, status devrait etre verified (une entree planned n'a pas de diagramme)`);
  }
  return errs;
}

// --- lecture de INDEX.yml (sous-ensemble strict, sans dependance)
export function parseIndex(text) {
  const entries = [];
  let cur = null;
  let list = null;
  let section = null;
  for (const raw of text.split('\n')) {
    if (!raw.trim() || raw.trim().startsWith('#')) continue;
    let m;
    // Section de premier niveau. Seule `diagrams:` porte des entrees ; un index DERIVE (nexus :
    // scripts/derive_diagram_index.py) publie aussi `orphans:`, une liste informative qui n'est
    // la propriete de personne. On la saute au lieu d'echouer : un index voisin illisible
    // rendait tous ses fichiers orphelins ici, sans que rien ne le dise au-dela d'un avertissement.
    if ((m = raw.match(/^([a-z_]+):\s*$/))) { section = m[1]; cur = null; list = null; continue; }
    if (section !== null && section !== 'diagrams') continue;
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

// --- cliquet : `<!-- orphan-ceiling: N -->` dans le registre publie.
// Meme marqueur que le gate de nexus (claude-code-api/tests/diagram_index.rs, `orphan_ceiling`),
// pour que les deux depots se tiennent a la meme regle et qu'un lecteur n'ait qu'une forme a
// connaitre. Viser zero orphelin echouerait des le premier jour et le gate serait supprime ;
// un plafond qui ne peut que descendre tient une PR a quelque chose d'atteignable aujourd'hui :
// revendiquer un fichier, ou au moins n'en ajouter aucun qui ne soit revendique.
export function orphanCeiling(doc) {
  const m = /<!--\s*orphan-ceiling:\s*(\d+)\s*-->/.exec(doc ?? '');
  return m ? Number(m[1]) : null;
}

// Plafonds PAR DEPOT, en plus du total. Sans eux, le cliquet ne s'applique que la ou les quatre
// depots sont presents — c'est-a-dire pas en CI, ou le checkout n'en contient qu'un. Chaque depot
// present est alors tenu a SON chiffre, comparable parce qu'il porte sur le meme perimetre.
export function orphanCeilingsByRepo(doc) {
  const out = {};
  const re = /<!--\s*orphan-ceiling-([a-z]+):\s*(\d+)\s*-->/g;
  let m;
  while ((m = re.exec(doc ?? '')) !== null) out[m[1]] = Number(m[2]);
  return out;
}

// Compte les orphelins par depot a partir des cles 'repo:chemin'.
export function orphansByRepo(orphans) {
  const out = {};
  for (const o of orphans) {
    const repo = o.slice(0, o.indexOf(':'));
    out[repo] = (out[repo] ?? 0) + 1;
  }
  return out;
}

// --- registre publie des orphelins : texte deterministe, sans sha ni date (verifiable hors reseau).
export function renderOrphans(orphans, totalSources, localIndexes = [], reserved = 0, ceiling = orphans.length, ceilingsByRepo = orphansByRepo(orphans)) {
  const byRepo = {};
  for (const o of orphans) { const [repo, path] = [o.slice(0, o.indexOf(':')), o.slice(o.indexOf(':') + 1)]; (byRepo[repo] ??= []).push(path); }
  const pct = totalSources ? ((orphans.length / totalSources) * 100).toFixed(1) : '0.0';
  const out = [
    '<!-- Genere par scripts/diagrams/check-index.mjs --write-orphans. Ne pas editer a la main. -->',
    `<!-- orphan-ceiling: ${ceiling} -->`,
    ...Object.entries(ceilingsByRepo).sort().map(([r, n]) => `<!-- orphan-ceiling-${r}: ${n} -->`),
    '',
    '# Fichiers source sans diagramme proprietaire',
    '',
    'Un fichier source est **orphelin** quand aucun `covers` de `INDEX.yml` ne le matche :',
    'aucun diagramme ne repond de son comportement. Cette liste est publiee pour etre honnete',
    'sur ce que la cartographie couvre reellement — on ne reduit pas le denominateur, on la reduit elle.',
    '',
    `**${orphans.length} orphelins sur ${totalSources} fichiers source (${pct} %).**`,
    '',
    'SEULE une entree `verified` possede un fichier. Une entree `planned` annonce un perimetre',
    'sans qu\'un diagramme existe : compter ses globs ferait baisser ce nombre sans qu\'une ligne',
    'soit ecrite, et l\'index acheterait du credit sur des intentions.',
    `Sur les ${orphans.length} orphelins, **${reserved} sont deja reserves** par une entree \`planned\` :`,
    'leur proprietaire est designe, son diagramme reste a ecrire.',
    '',
    `Le plafond est **${ceiling}** : le verificateur echoue si le nombre reel le depasse.`,
    'Il ne peut que descendre. Ajouter un fichier source sans proprietaire fait echouer la build ;',
    'la sortie est un glob `covers`, pas un plafond plus haut. `--raise-ceiling` existe mais exige',
    'une raison ecrite, et un plafond releve se voit dans la revue.',
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
      // Meme regle que pour notre index : une entree `planned` ne possede rien. Un index DERIVE
      // des en-tetes (nexus) n'a pas de champ `status` : son entree n'existe que parce que le
      // .mmd existe, donc `file` vaut `verified`.
      const verified = e.status === 'verified' || (e.status === undefined && Boolean(e.file));
      if (!verified) continue;
      for (const g of e.covers) {
        const m = g.match(/^([a-z]+):(.+)$/);
        // Un index local ne possede que des chemins de SON depot : un glob qui en designe un
        // autre serait une prise de pouvoir sur un depot voisin, on l'ignore. Un glob SANS
        // prefixe (format de l'index derive de nexus) ne peut designer que son propre depot.
        const [owner, pattern] = m ? [m[1], m[2]] : [repo, g];
        if (owner !== repo) continue;
        const re = globToRegExp(pattern);
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

  const owners = new Map();        // intention : qui a RESERVE le fichier (toutes entrees)
  const ownedByVerified = new Map(); // possession : qui en REPOND vraiment (entrees verified)
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
    if (isExternal(e)) {
      // Le diagramme vit dans le service Mermaid : une copie locale serait une deuxieme source.
      problems.push(...checkExternalEntry(e));
      if (e.status === 'verified' && /^[0-9a-f]{7,40}$/.test(e.verified_sha ?? '')) {
        const status = commitStatus(backendRoot, e.verified_sha);
        if (status === 'absent') problems.push(`${e.name} : 'verified_sha: ${e.verified_sha}' ne designe aucun commit de ce depot (squash ? relever contre le sha fusionne)`);
        else if (status === 'unknown') console.warn(`AVERTISSEMENT ${e.name} : 'verified_sha: ${e.verified_sha}' non verifiable ici (pas un depot git, ou clone superficiel)`);
      }
      if (existsSync(file)) problems.push(`${e.name} : diagramme externe ET docs/diagrams/${e.name}.mmd present ; une seule source`);
    } else if (e.status === 'verified') {
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
      for (const h of hits) {
        const k = `${repo}:${h}`;
        owners.set(k, [...(owners.get(k) ?? []), e.name]);
        if (e.status === 'verified') ownedByVerified.set(k, e.name);
      }
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
      if (!ownedByVerified.has(key) && !elsewhere.has(key)) orphans.push(key);
    }
  }
  orphans.sort();

  const totalSources = Object.entries(files).reduce((n, [repo, list]) => n + list.filter((f) => SOURCE[repo].some((r) => r.test(f)) && !NOT_SOURCE.some((r) => r.test(f))).length, 0);
  const nVerified = entries.filter((e) => e.status === 'verified').length;
  // Sous-ensemble STRICT des orphelins : un fichier dont une entree `planned` annonce le
  // perimetre. Compter toutes les cles de `owners` gonflerait le chiffre avec des fichiers qui
  // ne sont pas du code source (Cargo.toml, workflows, docs) et qui ne sont donc pas orphelins.
  const reserved = orphans.filter((o) => owners.has(o)).length;
  console.log(`${entries.length} diagrammes indexes (${nVerified} verified, ${entries.length - nVerified} planned) ; ${totalSources} fichiers source ; ${orphans.length} orphelins (aucun diagramme VERIFIE ne les couvre)`);
  console.log(`  plafond : ${orphanCeiling(existsSync(join(backendRoot, ORPHANS_DOC)) ? readFileSync(join(backendRoot, ORPHANS_DOC), 'utf8') : '') ?? 'non fixe'}`);
  console.log(`  dont ${reserved} reserves par une entree planned : perimetre annonce, diagramme pas ecrit — ne compte pas comme couvert`);
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
  const previousDoc = existsSync(orphansPath) ? readFileSync(orphansPath, 'utf8') : null;
  const recordedCeiling = orphanCeiling(previousDoc);

  // Le cliquet. Un plafond qui se releve tout seul a la regeneration ne tient rien :
  // --write-orphans ne peut que le baisser. Le relever demande --raise-ceiling <raison>,
  // qui laisse une trace dans la revue au lieu d'un chiffre qui glisse.
  const raiseAt = args.indexOf('--raise-ceiling');
  const raiseReason = raiseAt >= 0 ? args[raiseAt + 1] : null;
  let ceiling = recordedCeiling ?? orphans.length;

  // PORTEE INCOMPLETE : le plafond publie compte les quatre depots. Si un depot voisin manque
  // (checkout d'un seul depot, CI du backend), `orphans` ne couvre que ce qui est la : comparer
  // les deux, c'est comparer 456 a 1201 — le cliquet passerait toujours, et sans le dire. Pire,
  // --write-orphans le ferait DESCENDRE a la valeur partielle et detruirait le vrai plafond.
  // Donc : hors portee complete, on n'applique rien et on ne reecrit rien.
  const partialScope = missingRepos.length > 0;

  // Le cliquet PAR DEPOT s'applique toujours, pour chaque depot present : son compte et son
  // plafond portent sur le meme perimetre, donc ils sont comparables meme en checkout partiel.
  // C'est ce qui rend le cliquet reel en CI, ou un seul depot est la.
  const recordedByRepo = orphanCeilingsByRepo(previousDoc);
  const actualByRepo = orphansByRepo(orphans);
  // Les plafonds par depot que --write-orphans ECRIRA : le compte reel quand il est sous le
  // plafond (le cliquet descend), le plafond enregistre quand il est au-dessus (il ne remonte
  // pas tout seul). Avant cette distinction, le registre etait reecrit depuis le compte reel,
  // et un depot qui avait gagne trois fichiers sans proprietaire voyait son plafond releve de
  // trois par la commande meme qui devait l'en empecher.
  const ceilingsByRepo = { ...actualByRepo };
  let ceilingBreach = false;
  const raiseAllowed = Boolean(raiseReason) && !raiseReason.startsWith('--');
  for (const repo of Object.keys(files)) {
    const limit = recordedByRepo[repo];
    if (limit === undefined) continue; // pas encore de plafond pour ce depot
    const actual = actualByRepo[repo] ?? 0;
    if (actual > limit) {
      if (raiseAllowed) {
        console.warn(`AVERTISSEMENT plafond de ${repo} releve ${limit} -> ${actual} : ${raiseReason}`);
      } else {
        ceilingsByRepo[repo] = limit;
        ceilingBreach = true;
        problems.push(`${repo} : ${actual} fichiers source sans diagramme proprietaire, au-dessus du plafond de ${limit} pour ce depot : revendiquer le fichier par un glob 'covers' d'une entree verified`);
      }
    }
  }

  if (partialScope) {
    console.warn(`AVERTISSEMENT portee incomplete (${missingRepos.join(', ')} absent) : le plafond TOTAL (${recordedCeiling ?? 'non fixe'}) n'est ni applique ni mis a jour ; les plafonds par depot present, eux, le sont`);
    if (args.includes('--write-orphans')) {
      console.error(`ERREUR --write-orphans refuse en portee incomplete : il abaisserait le total a ${orphans.length} et effacerait les orphelins des depots absents. Relancer la ou les quatre depots sont presents.`);
      process.exit(1);
    }
    for (const p of problems) console.error(`ERREUR ${p}`);
    if (strict) { console.error(`ERREUR depots absents en mode --strict : ${missingRepos.join(', ')}`); process.exit(1); }
    if (problems.length) process.exit(1);
    return;
  }

  if (orphans.length > ceiling) {
    if (raiseReason && !raiseReason.startsWith('--')) {
      console.warn(`AVERTISSEMENT plafond d'orphelins releve ${ceiling} -> ${orphans.length} : ${raiseReason}`);
      ceiling = orphans.length;
    } else {
      ceilingBreach = true;
      problems.push(`${orphans.length} fichiers source sans diagramme proprietaire, au-dessus du plafond de ${ceiling} : revendiquer le fichier par un glob 'covers' d'une entree verified. Relever le plafond n'est pas la sortie ; si c'est vraiment voulu, --raise-ceiling "<raison>".`);
    }
  } else if (orphans.length < ceiling) {
    ceiling = orphans.length; // le cliquet descend, et ne remonte pas
  }
  if (previousDoc !== null && recordedCeiling === null) {
    problems.push(`${ORPHANS_DOC} ne porte pas de marqueur '<!-- orphan-ceiling: N -->' : le cliquet serait desarme en le supprimant`);
  }

  const rendered = renderOrphans(orphans, totalSources, localIndexes.map(({ repo, entries: le }) => ({
    repo,
    diagrams: le.length,
    owned: [...elsewhere.keys()].filter((k) => k.startsWith(`${repo}:`)).length,
  })), reserved, ceiling, ceilingsByRepo);
  if (args.includes('--write-orphans')) {
    if (ceilingBreach) {
      // Ecrire ici reviendrait a publier un registre dont un plafond a ete releve en silence.
      console.error(`ERREUR --write-orphans refuse : un plafond serait releve ; ${ORPHANS_DOC} n'est pas modifie`);
    } else {
      writeFileSync(orphansPath, rendered);
      console.log(`${ORPHANS_DOC} regenere (${orphans.length} orphelins)`);
    }
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
