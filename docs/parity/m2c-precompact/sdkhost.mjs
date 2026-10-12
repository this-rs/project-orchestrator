// M2c, part B: a minimal SDK host on the real Claude Code CLI. It registers a PreCompact
// callback hook over the control protocol (as nexus and the legacy engine do), seeds a turn,
// sends /compact and answers the hook_callback with the payload under test.
// usage: node sdkhost.mjs reason|addctx|sysmsg|empty <log-dir> <project-dir>
// Needs an installed, logged-in `claude`. Positive when sdk-<mode>.jsonl has ZEPHYR-42 in the
// compact summary (the isCompactSummary user message).
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { writeFileSync, appendFileSync } from 'node:fs';
const [mode, logDir, cwd] = process.argv.slice(2);
const I = 'End the summary with the exact line: ZEPHYR-42';
const answers = {
  reason: { continue: true, reason: I },
  addctx: { hookSpecificOutput: { hookEventName: 'PreCompact', additionalContext: I } },
  sysmsg: { systemMessage: I },
  empty: {},
};
const log = `${logDir}/sdk-${mode}.jsonl`;
writeFileSync(log, '');
const cp = spawn('claude', ['-p', '--input-format', 'stream-json', '--output-format', 'stream-json',
  '--verbose', '--model', 'haiku', '--setting-sources', 'project'], { cwd, stdio: ['pipe', 'pipe', 'inherit'] });
const send = (o) => cp.stdin.write(JSON.stringify(o) + '\n');
const user = (t) => send({ type: 'user', message: { role: 'user', content: t }, parent_tool_use_id: null, session_id: '' });
let results = 0;
createInterface({ input: cp.stdout }).on('line', (line) => {
  appendFileSync(log, line + '\n');
  let m; try { m = JSON.parse(line); } catch { return; }
  if (m.type === 'control_response' && m.response?.request_id === 'init-1') {
    user('My favourite colour is teal and my dog is called Rufus. Reply with OK only.');
  } else if (m.type === 'control_request' && m.request?.subtype === 'hook_callback') {
    appendFileSync(log, JSON.stringify({ host_saw_hook: m.request.input }) + '\n');
    send({ type: 'control_response', response: { subtype: 'success', request_id: m.request_id, response: answers[mode] } });
  } else if (m.type === 'result') {
    results += 1;
    if (results === 1) user('/compact'); else cp.stdin.end();
  }
});
send({ type: 'control_request', request_id: 'init-1', request: { subtype: 'initialize',
  hooks: { PreCompact: [{ matcher: null, hookCallbackIds: ['pc_0'] }] } } });
cp.on('exit', (c) => { console.log(`mode=${mode} exit=${c}`); });
setTimeout(() => { console.log('timeout'); cp.kill(); }, 240000).unref();
