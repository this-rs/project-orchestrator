#!/usr/bin/env node
// @ts-nocheck
//
// Unit tests for pre-tool-use.cjs — tool filtering logic
//
// Run: node hooks/pre-tool-use.test.cjs
//

'use strict';

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) {
    passed++;
    process.stderr.write(`  ✅ ${msg}\n`);
  } else {
    failed++;
    process.stderr.write(`  ❌ FAIL: ${msg}\n`);
  }
}

// ============================================================================
// Replicate the tool filtering logic from pre-tool-use.cjs
// ============================================================================

const ACTIVATABLE_TOOLS = new Set(['Grep', 'Glob', 'Read', 'Bash', 'Edit', 'Write']);
const MCP_TOOL_PREFIX = 'mcp__';

function isActivatable(toolName) {
  return ACTIVATABLE_TOOLS.has(toolName) || toolName.startsWith(MCP_TOOL_PREFIX);
}

// ============================================================================
// Tests: Native tools
// ============================================================================

process.stderr.write('\n--- Native tools ---\n');
assert(isActivatable('Grep'),  'Grep is activatable');
assert(isActivatable('Glob'),  'Glob is activatable');
assert(isActivatable('Read'),  'Read is activatable');
assert(isActivatable('Bash'),  'Bash is activatable');
assert(isActivatable('Edit'),  'Edit is activatable');
assert(isActivatable('Write'), 'Write is activatable');

// ============================================================================
// Tests: Non-activatable native tools
// ============================================================================

process.stderr.write('\n--- Non-activatable native tools ---\n');
assert(!isActivatable('Task'),           'Task is NOT activatable');
assert(!isActivatable('TodoWrite'),      'TodoWrite is NOT activatable');
assert(!isActivatable('WebSearch'),      'WebSearch is NOT activatable');
assert(!isActivatable('WebFetch'),       'WebFetch is NOT activatable');
assert(!isActivatable('EnterPlanMode'),  'EnterPlanMode is NOT activatable');
assert(!isActivatable('NotebookEdit'),   'NotebookEdit is NOT activatable');
assert(!isActivatable('AskUserQuestion'),'AskUserQuestion is NOT activatable');
assert(!isActivatable(''),               'Empty string is NOT activatable');

// ============================================================================
// Tests: MCP mega-tools (all 19)
// ============================================================================

process.stderr.write('\n--- MCP mega-tools ---\n');
const mcpTools = [
  'project', 'plan', 'task', 'step', 'decision', 'constraint',
  'release', 'milestone', 'commit', 'note', 'workspace',
  'workspace_milestone', 'resource', 'component', 'chat',
  'feature_graph', 'code', 'admin', 'skill',
];

for (const tool of mcpTools) {
  const fullName = `mcp__project-orchestrator__${tool}`;
  assert(isActivatable(fullName), `${fullName} is activatable`);
}

// ============================================================================
// Tests: Other MCP servers (should also be activatable)
// ============================================================================

process.stderr.write('\n--- Other MCP prefixes ---\n');
assert(isActivatable('mcp__github__create_issue'),    'mcp__github tool is activatable');
assert(isActivatable('mcp__filesystem__read_file'),    'mcp__filesystem tool is activatable');
assert(isActivatable('mcp__some-server__some_tool'),   'mcp__some-server tool is activatable');

// ============================================================================
// Tests: Edge cases
// ============================================================================

process.stderr.write('\n--- Edge cases ---\n');
assert(!isActivatable('MCP__uppercase'),  'MCP__ uppercase is NOT activatable (case sensitive)');
assert(isActivatable('mcp__'),            'mcp__ bare prefix is activatable (starts with prefix)');
assert(!isActivatable('mcp_single'),      'mcp_ single underscore is NOT activatable');
assert(!isActivatable('xmcp__fake'),      'xmcp__ non-prefix is NOT activatable');

// ============================================================================
// Summary
// ============================================================================

process.stderr.write(`\n--- Results: ${passed} passed, ${failed} failed ---\n`);
process.exit(failed > 0 ? 1 : 0);
