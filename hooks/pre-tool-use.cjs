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
// HTTP client (zero-dependency, with timeout)
// ============================================================================

/**
 * POST JSON to the PO server with strict timeout.
 * Returns parsed response body or null on any failure.
 */
function postActivate(port, payload) {
  return new Promise((resolve) => {
    const data = JSON.stringify(payload);
    const timeoutId = setTimeout(() => {
      debug('Request timeout');
      req.destroy();
      resolve(null);
    }, HOOK_TIMEOUT_MS);

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

  // 5. Call the PO server
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

  // 6. Output the hook response (stdout only)
  const output = {
    hookSpecificOutput: {
      additionalContext: response.context,
    },
  };

  process.stdout.write(JSON.stringify(output) + '\n');
  debug(`Injected ${response.context.length} chars from skill: ${response.skill_name}`);
}

// Global error handler — NEVER block the agent
main().catch((err) => {
  debug(`Fatal error: ${err.message}`);
  process.exit(0);
});
