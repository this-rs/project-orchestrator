#!/usr/bin/env node
// @ts-nocheck
//
// Project Orchestrator — PreToolUse Hook for Claude Code
//
// Intercepts Grep, Glob, Read, and Bash tool calls to inject contextual
// knowledge from the Neural Skills system.
//
// Protocol:
//   stdin  → JSON { hookEventName, toolName, toolInput }
//   stdout → JSON { hookSpecificOutput: { additionalContext: "..." } }
//
// Constraints:
//   - CommonJS only (no ESM, no TypeScript)
//   - Zero npm dependencies (Node.js builtins only)
//   - Timeout 300ms (never block the agent)
//   - Exit 0 on ANY error (graceful failure)
//   - stdout reserved for hook output, stderr for debug only
//
// Config: reads .po-config from cwd or parent directories
//   { "project_id": "uuid", "port": 6600 }

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

// ============================================================================
// Configuration
// ============================================================================

const HOOK_TIMEOUT_MS = 300;
const CONFIG_FILENAME = '.po-config';
const DEFAULT_PORT = 6600;
const DEBUG = process.env.PO_HOOK_DEBUG === '1';

// Cache & throttle settings
const CACHE_TTL_MS = 30000;        // 30s TTL for cached responses
const MAX_PER_SKILL = 3;           // Max injections per skill per session
const MAX_GLOBAL = 10;             // Max total injections per session
const PPID = process.ppid;         // Claude Code parent process PID
const CACHE_FILE = path.join(require('os').tmpdir(), `po-hook-cache-${PPID}.json`);

// Tools that trigger skill activation
const ACTIVATABLE_TOOLS = new Set(['Grep', 'Glob', 'Read', 'Bash', 'Edit', 'Write']);

// ============================================================================
// Logging (stderr only, never stdout)
// ============================================================================

function debug(msg) {
  if (DEBUG) {
    process.stderr.write(`[PO-HOOK] ${msg}\n`);
  }
}

// ============================================================================
// Config discovery
// ============================================================================

/**
 * Find .po-config by walking up from cwd.
 * Returns parsed config or null if not found.
 */
function findConfig(startDir) {
  let dir = startDir || process.cwd();
  const root = path.parse(dir).root;

  for (let i = 0; i < 20; i++) { // safety limit
    const configPath = path.join(dir, CONFIG_FILENAME);
    try {
      const content = fs.readFileSync(configPath, 'utf-8');
      const config = JSON.parse(content);
      debug(`Config found: ${configPath}`);
      return config;
    } catch (_) {
      // Not found here, try parent
    }

    const parent = path.dirname(dir);
    if (parent === dir || parent === root) break;
    dir = parent;
  }

  return null;
}

// ============================================================================
// Cache & Throttle
// ============================================================================

/**
 * Read the cache file. Returns cache object or a fresh one.
 * Cache structure:
 * {
 *   ppid: number,                   // Parent PID (session identifier)
 *   entries: {                      // Keyed by skill_id
 *     [skill_id]: {
 *       context: string,            // Cached context
 *       timestamp: number,          // Date.now() when cached
 *       count: number,              // Injection count for this skill
 *     }
 *   },
 *   global_count: number,           // Total injections this session
 * }
 */
function readCache() {
  try {
    const raw = fs.readFileSync(CACHE_FILE, 'utf-8');
    const cache = JSON.parse(raw);
    // If ppid changed, this is a new session — start fresh
    if (cache.ppid !== PPID) {
      debug(`Session changed (cache ppid=${cache.ppid}, current=${PPID}), resetting cache`);
      return freshCache();
    }
    return cache;
  } catch (_) {
    return freshCache();
  }
}

function freshCache() {
  return { ppid: PPID, entries: {}, global_count: 0 };
}

function writeCache(cache) {
  try {
    fs.writeFileSync(CACHE_FILE, JSON.stringify(cache), 'utf-8');
  } catch (e) {
    debug(`Failed to write cache: ${e.message}`);
  }
}

/**
 * Check if we have a valid cached response for this skill.
 * Returns the cached context string, or null if miss/expired.
 */
function getCachedContext(cache, skillId) {
  const entry = cache.entries[skillId];
  if (!entry) return null;

  const age = Date.now() - entry.timestamp;
  if (age > CACHE_TTL_MS) {
    debug(`Cache expired for skill ${skillId} (age=${age}ms)`);
    delete cache.entries[skillId];
    return null;
  }

  debug(`Cache hit for skill ${skillId} (age=${age}ms)`);
  return entry.context;
}

/**
 * Check if throttle limits have been reached.
 * Returns true if the hook should be suppressed.
 */
function isThrottled(cache, skillId) {
  // Global limit
  if (cache.global_count >= MAX_GLOBAL) {
    debug(`Global throttle reached (${cache.global_count}/${MAX_GLOBAL})`);
    return true;
  }

  // Per-skill limit
  const entry = cache.entries[skillId];
  if (entry && entry.count >= MAX_PER_SKILL) {
    debug(`Per-skill throttle reached for ${skillId} (${entry.count}/${MAX_PER_SKILL})`);
    return true;
  }

  return false;
}

/**
 * Record a successful injection in the cache.
 */
function recordInjection(cache, skillId, context) {
  const entry = cache.entries[skillId] || { context: '', timestamp: 0, count: 0 };
  entry.context = context;
  entry.timestamp = Date.now();
  entry.count += 1;
  cache.entries[skillId] = entry;
  cache.global_count += 1;
  writeCache(cache);
  debug(`Recorded injection: skill=${skillId} count=${entry.count} global=${cache.global_count}`);
}

// ============================================================================
// HTTP client (zero-dependency, with timeout)
// ============================================================================

/**
 * POST JSON to the PO server with strict timeout.
 * Returns parsed response body or null on any failure.
 */
function postActivate(port, payload) {
  return new Promise((resolve) => {
    const data = JSON.stringify(payload);

    const req = http.request(
      {
        hostname: '127.0.0.1',
        port: port,
        path: '/api/hooks/activate',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
        timeout: HOOK_TIMEOUT_MS,
      },
      (res) => {
        // 204 No Content → no skill matched
        if (res.statusCode === 204) {
          clearTimeout(timeoutId);
          debug('204 No Content — no skill matched');
          resolve(null);
          return;
        }

        // Non-200 → error
        if (res.statusCode !== 200) {
          clearTimeout(timeoutId);
          debug(`Server returned ${res.statusCode}`);
          resolve(null);
          return;
        }

        // Collect response body
        let body = '';
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => {
          clearTimeout(timeoutId);
          try {
            resolve(JSON.parse(body));
          } catch (_) {
            debug('Failed to parse response JSON');
            resolve(null);
          }
        });
      }
    );

    req.on('error', (err) => {
      clearTimeout(timeoutId);
      debug(`Request error: ${err.message}`);
      resolve(null);
    });

    req.on('timeout', () => {
      debug('Socket timeout');
      req.destroy();
      // resolve(null) will be called by the error handler
    });

    // Schedule timeout AFTER req is created (avoid TDZ)
    const timeoutId = setTimeout(() => {
      debug('Request timeout');
      req.destroy();
      resolve(null);
    }, HOOK_TIMEOUT_MS);

    req.write(data);
    req.end();
  });
}

// ============================================================================
// Stdin reader
// ============================================================================

/**
 * Read all of stdin synchronously.
 * Claude Code pipes the hook input as a single JSON blob.
 */
function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    let resolved = false;
    const done = (value) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(safetyTimer);
      resolve(value);
    };
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (chunk) => { data += chunk; });
    process.stdin.on('end', () => {
      try {
        done(JSON.parse(data));
      } catch (_) {
        done(null);
      }
    });
    // Safety timeout — if stdin never ends, bail out
    const safetyTimer = setTimeout(() => {
      debug('Stdin read timeout');
      done(null);
    }, 1000);
  });
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  // 1. Read hook input from stdin
  const input = await readStdin();
  if (!input) {
    debug('No input or invalid JSON');
    return;
  }

  const { hookEventName, toolName, toolInput } = input;

  // 2. Only handle PreToolUse events
  if (hookEventName !== 'PreToolUse') {
    debug(`Ignoring event: ${hookEventName}`);
    return;
  }

  // 3. Only handle activatable tools
  if (!ACTIVATABLE_TOOLS.has(toolName)) {
    debug(`Ignoring tool: ${toolName}`);
    return;
  }

  // 4. Find config (project_id, port)
  const config = findConfig(input.cwd || process.cwd());
  if (!config || !config.project_id) {
    debug('No .po-config found or missing project_id');
    return;
  }

  const port = config.port || DEFAULT_PORT;

  // 5. Load cache & check throttle
  const cache = readCache();

  // Check global throttle before making any server call
  if (cache.global_count >= MAX_GLOBAL) {
    debug(`Global throttle: ${cache.global_count}/${MAX_GLOBAL} — suppressing`);
    return;
  }

  // 6. Call the PO server
  const payload = {
    project_id: config.project_id,
    tool_name: toolName,
    tool_input: toolInput || {},
  };

  debug(`Activating: tool=${toolName} project=${config.project_id}`);

  const response = await postActivate(port, payload);
  if (!response || !response.context) {
    debug('No context returned');
    return;
  }

  // 7. Check per-skill throttle and cache
  const skillId = response.skill_id || response.skill_name || 'unknown';

  // Check if this skill is throttled
  if (isThrottled(cache, skillId)) {
    debug(`Skill ${skillId} throttled — suppressing output`);
    return;
  }

  // Check if we have a recent cached response for this skill
  const cachedCtx = getCachedContext(cache, skillId);
  if (cachedCtx) {
    // Use cached context instead of server response (saves token budget)
    const output = {
      hookSpecificOutput: {
        additionalContext: cachedCtx,
      },
    };
    process.stdout.write(JSON.stringify(output) + '\n');
    // Cache hits do NOT count toward throttle limits — only fresh server
    // responses should increment counters to prevent premature throttling.
    debug(`Injected cached context for skill: ${skillId}`);
    return;
  }

  // 8. Output the hook response (stdout only)
  const output = {
    hookSpecificOutput: {
      additionalContext: response.context,
    },
  };

  process.stdout.write(JSON.stringify(output) + '\n');

  // Record the injection in cache
  recordInjection(cache, skillId, response.context);
  debug(`Injected ${response.context.length} chars from skill: ${skillId}`);
}

// Global error handler — NEVER block the agent
main().catch((err) => {
  debug(`Fatal error: ${err.message}`);
  process.exit(0);
});
