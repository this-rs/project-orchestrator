#!/usr/bin/env node
// Publication helper for mermaid.ffs.dev. See README.md.
// Secrets: the passphrase is never read or handled here. Authentication uses a
// session cookie taken from MERMAID_COOKIE or the file named by MERMAID_COOKIE_FILE
// (keep that file outside the repository).
import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const BASE = process.env.MERMAID_URL || 'https://mermaid.ffs.dev';
// mermaid.min.js lives NEXT TO this script (not in the parent folder).
export const MERMAID_PATH = join(HERE, 'mermaid.min.js');
const require = createRequire(import.meta.url); // NODE_PATH is ignored in ESM

/** Cheap checks for the known linter traps (see note 47c540b9). */
export function preCheck(code) {
  const errors = [];
  code.split('\n').forEach((line, i) => {
    if (/^\s*%%\s*$/.test(line) && !/^\s*%% $/.test(line.replace(/\r$/, '')) )
      errors.push(`line ${i + 1}: bare "%%" comment; write "%% " with a trailing space`);
    if (/^\s*state\s+"[^"]*"\s+as\s+\w+/.test(line))
      errors.push(`line ${i + 1}: 'state "x" as Y' is refused; write 'Y : x'`);
  });
  return errors;
}

/** Parse with the exact mermaid bundle inside JSDOM. Returns {ok, error?}. */
export async function parseWithMermaid(code, mermaidPath = MERMAID_PATH) {
  if (!existsSync(mermaidPath))
    throw new Error(`mermaid.min.js not found at ${mermaidPath}; run: node publish.mjs fetch-vendor`);
  const { JSDOM } = require('jsdom');
  const dom = new JSDOM('<!doctype html><html><body></body></html>', { runScripts: 'dangerously' });
  try {
    const s = dom.window.document.createElement('script'); // window.eval can't see the IIFE global
    s.textContent = readFileSync(mermaidPath, 'utf8');
    dom.window.document.head.appendChild(s);
    const m = dom.window.__esbuild_esm_mermaid_nm?.mermaid ?? dom.window.mermaid;
    const mer = m?.default ?? m;
    if (!mer?.parse) throw new Error('mermaid global not found after loading bundle');
    try { await mer.parse(code); return { ok: true }; }
    catch (e) { return { ok: false, error: String(e?.message ?? e).split('\n')[0] }; }
  } finally { dom.window.close(); }
}

const NEGATIVE = 'flowchart TD\n  A --> \n  B[[[ ( ) ---> }}}\n';

/** Negative control: a broken diagram MUST be rejected, else "OK" means nothing. */
export async function negativeControl(mermaidPath = MERMAID_PATH) {
  const r = await parseWithMermaid(NEGATIVE, mermaidPath);
  if (r.ok) throw new Error('negative control FAILED: broken diagram was accepted by the parser');
}

export async function lintLocal(code, mermaidPath = MERMAID_PATH) {
  await negativeControl(mermaidPath);
  const errors = preCheck(code);
  if (errors.length) return { ok: false, errors };
  const r = await parseWithMermaid(code, mermaidPath);
  return r.ok ? { ok: true, errors: [] } : { ok: false, errors: [r.error] };
}

function cookie() {
  if (process.env.MERMAID_COOKIE) return process.env.MERMAID_COOKIE.trim();
  const f = process.env.MERMAID_COOKIE_FILE;
  if (f && existsSync(f)) return readFileSync(f, 'utf8').trim();
  throw new Error('no session: set MERMAID_COOKIE or MERMAID_COOKIE_FILE (file outside the repo)');
}

async function api(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: { 'content-type': 'application/json', cookie: cookie() },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let json; try { json = JSON.parse(text); } catch { json = { raw: text }; }
  if (!res.ok || json.error) throw new Error(`${method} ${path} -> ${res.status} ${JSON.stringify(json)}`);
  return json;
}

async function gate(code) {
  const l = await lintLocal(code);
  if (!l.ok) throw new Error('local lint failed:\n  ' + l.errors.join('\n  '));
  const s = await api('POST', '/lint', { code });
  if (!s.valid) throw new Error('server lint failed: ' + (s.summary ?? JSON.stringify(s)));
}

function flags(argv) {
  const o = {}; const rest = [];
  for (let i = 0; i < argv.length; i++)
    argv[i].startsWith('--') ? (o[argv[i].slice(2)] = argv[++i]) : rest.push(argv[i]);
  return [o, rest];
}

async function main() {
  const [cmd, ...argv] = process.argv.slice(2);
  const [f, rest] = flags(argv);
  switch (cmd) {
    case 'fetch-vendor': { // public endpoint, no login needed
      const r = await fetch(BASE + '/vendor/mermaid.min.js');
      if (!r.ok) throw new Error(`HTTP ${r.status}; use the npm "mermaid" package instead`);
      writeFileSync(MERMAID_PATH, Buffer.from(await r.arrayBuffer()));
      console.log('saved', MERMAID_PATH); break;
    }
    case 'lint': { // local only, no session needed
      const l = await lintLocal(readFileSync(rest[0], 'utf8'));
      if (!l.ok) { console.error('FAIL\n  ' + l.errors.join('\n  ')); process.exit(1); }
      console.log('OK (negative control passed)'); break;
    }
    case 'create': {
      const code = readFileSync(rest[0], 'utf8'); await gate(code);
      const name = /^%%\s*name:\s*(.+)$/m.exec(code)?.[1]?.trim();
      console.log(JSON.stringify(await api('POST', '/create',
        { name, code, workspace: f.workspace, session: f.session, strict: true }))); break;
    }
    case 'update': {
      const code = readFileSync(rest[1], 'utf8'); await gate(code);
      console.log(JSON.stringify(await api('POST', '/update', { id: rest[0], code, by: f.by || 'agent' }))); break;
    }
    case 'archive': // UNVERIFIED endpoint: not documented in note 47c540b9; confirm with a logged-in session.
      console.log(JSON.stringify(await api('POST', '/archive', { id: rest[0] }))); break;
    default:
      console.error('usage: publish.mjs fetch-vendor | lint <f> | create <f> [--workspace W --session S] | update <id> <f> [--by who] | archive <id>');
      process.exit(2);
  }
}
if (process.argv[1] === fileURLToPath(import.meta.url))
  main().catch((e) => { console.error(String(e.message || e)); process.exit(1); });
