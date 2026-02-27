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
// Replicate extractFilePath from pre-tool-use.cjs
// ============================================================================

function extractFilePath(toolName, toolInput) {
  if (!toolInput || typeof toolInput !== 'object') return null;

  if (toolInput.file_path && typeof toolInput.file_path === 'string' && toolInput.file_path.startsWith('/')) {
    return toolInput.file_path;
  }

  if (toolInput.path && typeof toolInput.path === 'string' && toolInput.path.startsWith('/')) {
    return toolInput.path;
  }

  if (toolName === 'Bash' && toolInput.command && typeof toolInput.command === 'string') {
    const pathMatch = toolInput.command.match(/(?:^|\s)(\/[^\s;|&>"']+)/);
    if (pathMatch) return pathMatch[1];
  }

  if (toolName.startsWith('mcp__')) {
    for (const field of ['file_path', 'path', 'target', 'node_path', 'root_path', 'cwd']) {
      const val = toolInput[field];
      if (val && typeof val === 'string' && val.startsWith('/')) {
        return val;
      }
    }
  }

  return null;
}

// ============================================================================
// Tests: extractFilePath
// ============================================================================

process.stderr.write('\n--- extractFilePath: native tools ---\n');
assert(
  extractFilePath('Read', { file_path: '/Users/dev/src/main.rs' }) === '/Users/dev/src/main.rs',
  'Read extracts file_path'
);
assert(
  extractFilePath('Edit', { file_path: '/Users/dev/src/lib.rs', old_string: 'a', new_string: 'b' }) === '/Users/dev/src/lib.rs',
  'Edit extracts file_path'
);
assert(
  extractFilePath('Write', { file_path: '/tmp/output.txt', content: 'hello' }) === '/tmp/output.txt',
  'Write extracts file_path'
);
assert(
  extractFilePath('Grep', { pattern: 'foo', path: '/Users/dev/src' }) === '/Users/dev/src',
  'Grep extracts path'
);
assert(
  extractFilePath('Grep', { pattern: 'foo' }) === null,
  'Grep without path returns null'
);
assert(
  extractFilePath('Read', { file_path: 'relative/path.rs' }) === null,
  'Relative file_path returns null'
);

process.stderr.write('\n--- extractFilePath: Bash ---\n');
assert(
  extractFilePath('Bash', { command: 'cargo test /Users/dev/src/test.rs' }) === '/Users/dev/src/test.rs',
  'Bash extracts absolute path from command'
);
assert(
  extractFilePath('Bash', { command: 'ls -la' }) === null,
  'Bash without absolute path returns null'
);
assert(
  extractFilePath('Bash', { command: 'cat /etc/passwd' }) === '/etc/passwd',
  'Bash extracts path after command'
);

process.stderr.write('\n--- extractFilePath: MCP tools ---\n');
assert(
  extractFilePath('mcp__project-orchestrator__code', { action: 'get_file_symbols', file_path: '/Users/dev/src/main.rs' }) === '/Users/dev/src/main.rs',
  'MCP code.get_file_symbols extracts file_path'
);
assert(
  extractFilePath('mcp__project-orchestrator__code', { action: 'analyze_impact', target: '/Users/dev/src/api.rs' }) === '/Users/dev/src/api.rs',
  'MCP code.analyze_impact extracts target'
);
assert(
  extractFilePath('mcp__project-orchestrator__admin', { action: 'sync_directory', path: '/Users/dev/project' }) === '/Users/dev/project',
  'MCP admin.sync_directory extracts path'
);
assert(
  extractFilePath('mcp__project-orchestrator__code', { action: 'get_node_importance', node_path: '/Users/dev/src/lib.rs' }) === '/Users/dev/src/lib.rs',
  'MCP code.get_node_importance extracts node_path'
);
assert(
  extractFilePath('mcp__project-orchestrator__chat', { action: 'send_message', cwd: '/Users/dev/project' }) === '/Users/dev/project',
  'MCP chat.send_message extracts cwd'
);
assert(
  extractFilePath('mcp__project-orchestrator__task', { action: 'create', title: 'foo' }) === null,
  'MCP task.create without path returns null'
);

process.stderr.write('\n--- extractFilePath: edge cases ---\n');
assert(extractFilePath('Read', null) === null, 'null toolInput returns null');
assert(extractFilePath('Read', 'string') === null, 'string toolInput returns null');
assert(extractFilePath('Read', {}) === null, 'empty object returns null');
assert(
  extractFilePath('Read', { file_path: '' }) === null,
  'empty file_path returns null'
);

// ============================================================================
// Summary
// ============================================================================

process.stderr.write(`\n--- Results: ${passed} passed, ${failed} failed ---\n`);
process.exit(failed > 0 ? 1 : 0);
