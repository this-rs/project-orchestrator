# Chat & WebSocket Guide

Real-time chat with Claude and live CRUD notifications via WebSocket connections.

---

## Overview

Project Orchestrator provides two WebSocket endpoints for real-time communication:

1. **Chat WebSocket** (`/ws/chat/{session_id}`) -- Bidirectional chat with Claude, replacing the previous SSE-based streaming approach with a single WebSocket connection per client per session.
2. **CRUD Events WebSocket** (`/ws/events`) -- Real-time notifications for all create, update, delete, link, and unlink operations across the system.

The chat system is built on the [nexus-claude SDK](https://github.com/anthropics/nexus-claude) with full memory support. Conversations are session-based and persisted in Neo4j, allowing resume and replay.

---

## Architecture

```
Client <--> WebSocket <--> ChatManager <--> nexus-claude SDK <--> Claude
                                |                    |
                           Neo4j (sessions,     NATS (cross-instance
                            messages, events)    chat relay, events)
```

- **ChatManager** manages active sessions, broadcast channels, and the nexus-claude SDK integration
- **Neo4j** persists session metadata, message history, and structured chat events for replay
- **Meilisearch** indexes messages for full-text search across sessions
- **NATS** enables cross-instance chat relay (RPC send, interrupt forwarding, streaming snapshots) and CRUD event sync
- **EventBus** broadcasts CRUD events to all connected `/ws/events` clients (local + NATS hybrid)

---

## Chat WebSocket (`/ws/chat/{session_id}`)

### Connection

Connect to an existing session by its UUID. The session must already exist in Neo4j (created via `POST /api/chat/sessions` or the `chat_send_message` MCP tool).

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/chat/SESSION_UUID');
```

#### Authentication Handshake

The first message sent over the WebSocket **must** be an authentication message. The server waits up to 10 seconds for it.

```javascript
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "auth",
    token: "eyJhbGciOiJIUzI1NiIs..."
  }));
};
```

The server responds with either:

```json
{"type": "auth_ok", "user": {"id": "uuid", "email": "user@example.com", "name": "Alice"}}
```

or on failure:

```json
{"type": "auth_error", "message": "Invalid token: ..."}
```

After `auth_ok`, the server automatically replays persisted events and begins forwarding live events.

#### Query Parameters

| Parameter | Description |
|-----------|-------------|
| `last_event` | Last event sequence number seen by the client (for partial replay). Default: `0` |

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/chat/SESSION_UUID?last_event=42');
```

### Connection Lifecycle

1. **Connect** -- WebSocket upgrade accepted
2. **Auth** -- Client sends `{"type": "auth", "token": "..."}`, server validates JWT
3. **Replay** -- Server replays persisted events since `last_event` (with `"replaying": true` flag)
4. **Streaming snapshot** -- If Claude is currently streaming, accumulated events and partial text are sent
5. **`replay_complete`** -- Marker event signaling replay is finished
6. **Live events** -- Real-time events forwarded from the broadcast channel
7. **Ping/Pong** -- Server sends pings every 30 seconds to detect dead clients

### Sending Messages (Client to Server)

All client messages are JSON with a `type` field:

#### `user_message` -- Send a chat message

```json
{"type": "user_message", "content": "Search for authentication code"}
```

If the session is not currently active (no CLI process running), the server automatically resumes it.

#### `interrupt` -- Interrupt the current operation

```json
{"type": "interrupt"}
```

A delivered interrupt is followed by a watchdog. If the turn is still streaming
5 s later (its task never read the cancellation: a store call that never
answered, for instance), the server abandons the turn itself: the task is
aborted, `streaming_status` goes to `false` and an `error` event with
`code: "turn_abandoned"` says why. Messages held during the turn stay queued
and go out after the next one. Events the turn had already handed to the
store are still written; if it was stopped before its answer was complete,
part of that answer may be missing after a reload.

Each post-stream step (memory, auto-continue, objective reminder) runs under
a 30 s budget, and so does the wait for the turn's own events to be saved. A
step that overruns is skipped: `error` with `code: "post_stream_step_abandoned"`,
`reason` naming the step. A save that overruns is not dropped: it keeps running
and lands when the database answers, the turn goes on (queued messages are
sent), and `error` with `code: "persistence_delayed"` (`reason`: the step, e.g.
`persist_events`) tells the client that a reload may not show the latest
messages yet. Both are also logged by the server with the step name.

#### `permission_response` -- Respond to a permission request

```json
{"type": "permission_response", "id": "pr_1", "allow": true, "scope": "session"}
```

`scope` (optional, default `once`) says how long an approval lasts: `once` (this call) or
`session`. Both engines declare what they accept in `system_init.capabilities.permission_scopes`
(the Claude Code engine: `["once", "session"]`, in the whole Claude Code capability object, as
the agent engine stamps it); a scope not declared is refused. A `session`
approval is decided by the BACKEND (`chat::session_grants`), never handed to the provider as a
rule (no nexus `allow` entry, no `updatedPermissions` to the CLI): the provider is answered
`once`, and the backend answers itself the later requests of the SAME session that the grant
covers. A grant only ever covers exactly what the user saw and approved, and only for an action
that cannot run code the model can change. Everything is an allowlist; anything else is refused
(`permission_scope_unsupported`, the user answers `once`):

- **the request must carry the whole call.** The Claude Code engine and the native engine hand
  the model's own tool input. On Codex, only an MCP tool call whose elicitation carries its
  arguments (`tool_params`) qualifies. Never Codex's `apply_patch` (its input is `{reason}`: no
  path, no diff, so one grant would cover every later patch), its `shell` (nexus joins the argv
  with spaces: `["rm","a b"]` and `["rm","a","b"]` read the same) or `request_permissions`;
  never an ACP agent;
- **any call of a read-only built-in tool of the native engine** (`mcp__nexus__Read`, `Glob`,
  `Grep`, `LS`, `NotebookRead`), identified by the adapter, never by the name's suffix;
- **the identical call** (same tool, same input; a command's surrounding blanks trimmed, its
  `description` ignored, every other field compared) for: the file tools (`Write`, `Edit`,
  `MultiEdit`, `NotebookEdit`), a read of the Claude Code CLI (it asks a read only outside the
  working directory, so never the whole tool), `WebFetch`, `WebSearch`, a third party's MCP tool
  (`mcp__acme__Read` included);
- **the identical command line** (`Bash`, `Monitor`, or a tool named like one) only when every
  simple command of it (split on `; & | ( )` and new lines) runs one of these programs, plainly
  named: `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `egrep`, `fgrep`, `rg` (without `--pre`,
  `--pre-glob`, `--hostname-bin`), `find` (without `-exec`, `-execdir`, `-ok`, `-okdir`), `fd` /
  `fdfind` (without `-x`, `-X`, `--exec`, `--exec-batch`), `sort` (without
  `--compress-program`), `uniq`, `cut`, `tr`, `nl`, `diff`, `cmp`, `stat`, `file`, `du`, `df`,
  `tree`, `pwd`, `echo`, `printf` (without `-v`), `which`, `basename`, `dirname`, `realpath`,
  `readlink`, `whoami`, `uname`, `id`, `true`, `false`. A forbidden long option is refused
  abbreviated too (`--compress`), and an unquoted glob is refused next to a program that has one
  (a file named `--pre=./x.sh`). Everything else is refused: interpreters, shells, scripts given
  by path, task runners and build tools, tools that load project files (`eslint`, `vite`,
  `mypy`...), `git` (hooks, `core.fsmonitor`, diff drivers), `sed`, `awk`, `jq`, `tar`,
  `sqlite3`, `xargs`, `env`, `sudo`, any builtin that changes how a name resolves (`export`,
  `hash`, `enable`, `alias`, `cd`...), an assignment in front (`PATH=./bin ls`);
- **a line that cannot be read for sure** gets no grant: an expansion (`$` outside single quotes,
  a backquote, `<(...)`), a brace outside quotes (`{bash,x.sh}` runs `bash x.sh`), a backslash
  outside single quotes (`ba\⏎sh x.sh` is a line continuation), an unterminated quote, a control
  character;
- **never** `SlashCommand`, `Skill` (they run a file the model can edit), `Task` / `Agent`,
  `TaskStop`, or a tool the backend does not know.

The working directory is **not** part of the match: both engines keep the `cd` of one call for
the next, so a grant of `ls` lists whatever directory the shell is in, and a grant of
`echo x > out` writes `out` there. The programs are found through the server's `PATH` (neither
engine keeps an environment change from one call to the next).

Only requests the provider asked reach the backend, after its own policy (read-only access,
denies, trust) and the project's consent. Another session, or the same one after a restart, asks
again. `always` is part of the contract but refused on every engine for now (P11b). A scope that
cannot be kept is refused with an `error` frame `{"code": "permission_scope_unsupported",
"reason": "<scope>"}`; nothing is answered and the request stays waiting. An answer to a request
that no longer waits (answered by the backend under a grant, or from another tab) is ignored,
on the WebSocket as on REST and NATS. The resulting `permission_decision` carries `scope` and
`rule` (what a session grant covers, e.g. `Bash: ls -la`). The REST twin
`POST /api/chat/sessions/{id}/permissions/{request_id}` takes `{"allow", "scope"?}` (400
`permission_scope_unsupported`); it is a human route (an agent session token gets 403).

#### `input_response` -- Respond to an input request

```json
{"type": "input_response", "id": "ir_1", "content": "option B"}
```

### Server Event Types (12 types)

Events sent from the server to the client. Each event includes a `type` field and optionally a `seq` (sequence number) for replay ordering.

| Event | Description | Payload Fields |
|-------|-------------|----------------|
| `user_message` | Echo of the sent message (for multi-tab sync) | `content` |
| `assistant_text` | Text response chunk from Claude | `content` |
| `thinking` | Claude's extended thinking content | `content` |
| `tool_use` | Claude is invoking a tool | `id`, `tool`, `input` |
| `tool_result` | Result of a tool invocation | `id`, `result`, `is_error` |
| `tool_use_input_resolved` | Full input resolved for a tool_use (emitted when the complete input arrives after an initial empty one) | `id`, `input` |
| `permission_request` | Claude needs permission to use a tool | `id`, `tool`, `input` |
| `ask_user_question` | Claude asks the user a question (AskUserQuestion tool) | `id`, `tool_call_id`, `questions`, `input` |
| `result` | Conversation turn completed | `session_id`, `duration_ms`, `cost_usd` (optional) |
| `stream_delta` | Raw streaming text token (real-time) | `text` |
| `streaming_status` | Stream state change | `is_streaming` (boolean) |
| `error` | An error occurred | `message`, `code` (optional, e.g. `turn_abandoned`, `persistence_delayed`, `post_stream_step_abandoned`, `refs_invalid`), `reason` (optional) |

#### Special Control Events

These events are not `ChatEvent` variants but are sent by the WebSocket handler for connection management:

| Event | Description |
|-------|-------------|
| `auth_ok` | Authentication succeeded |
| `auth_error` | Authentication failed |
| `replay_complete` | All persisted events have been replayed |
| `partial_text` | Accumulated stream_delta text snapshot (mid-stream join) |
| `events_lagged` | Client fell behind the broadcast; some events were skipped |
| `session_closed` | The session was cleaned up on the server |

#### Example Event

```json
{
  "type": "tool_use",
  "id": "tu_abc123",
  "tool": "search_code",
  "input": {"query": "authenticate", "project_slug": "my-api"},
  "seq": 15
}
```

#### Replay Events

During the replay phase, events include `"replaying": true` so the client can distinguish replayed history from live events:

```json
{
  "type": "assistant_text",
  "content": "I found the authentication module.",
  "seq": 3,
  "replaying": true
}
```

---

## CRUD Events WebSocket (`/ws/events`)

Real-time notifications for all CRUD operations across the system. Useful for building live dashboards and keeping UIs in sync.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/events');

// Authenticate (required)
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "auth",
    token: "eyJhbGciOiJIUzI1NiIs..."
  }));
};
```

### Query Parameters (Filtering)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `entity_types` | Comma-separated entity types to subscribe to | `task,plan,note` |
| `project_id` | Filter events by project UUID | `550e8400-...` |

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/events?entity_types=task,plan&project_id=UUID');
```

Events without a `project_id` (global events) always pass through the project filter.

### Entity Types (15)

| Entity Type | Serialized Name |
|-------------|-----------------|
| Project | `project` |
| Plan | `plan` |
| Task | `task` |
| Step | `step` |
| Decision | `decision` |
| Constraint | `constraint` |
| Commit | `commit` |
| Release | `release` |
| Milestone | `milestone` |
| Workspace | `workspace` |
| WorkspaceMilestone | `workspace_milestone` |
| Resource | `resource` |
| Component | `component` |
| Note | `note` |
| ChatSession | `chat_session` |

### Actions

| Action | Description |
|--------|-------------|
| `created` | A new entity was created |
| `updated` | An existing entity was modified |
| `deleted` | An entity was removed |
| `linked` | A relationship was created between two entities |
| `unlinked` | A relationship was removed between two entities |

### Event Format

```json
{
  "entity_type": "task",
  "action": "updated",
  "entity_id": "550e8400-e29b-41d4-a716-446655440000",
  "related": {
    "entity_type": "release",
    "entity_id": "660e8400-e29b-41d4-a716-446655440001"
  },
  "payload": {"status": "completed"},
  "project_id": "770e8400-e29b-41d4-a716-446655440002",
  "timestamp": "2026-02-10T14:30:00.000Z"
}
```

Notes:
- `related` is only present for `linked`/`unlinked` actions
- `payload` is omitted when null (e.g., for `deleted` events)
- `project_id` is omitted for global events (workspace-level operations)

---

## Chat Sessions

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat/sessions` | Create a new session and send the first message |
| GET | `/api/chat/sessions` | List sessions (optional `?project_slug=...` filter) |
| GET | `/api/chat/sessions/{id}` | Get session details |
| DELETE | `/api/chat/sessions/{id}` | Delete a session (closes active CLI process) |
| GET | `/api/chat/sessions/{id}/messages` | Get message history (paginated) |
| GET | `/api/chat/search?q=...` | Search messages across all sessions |
| POST | `/api/chat/sessions/backfill-previews` | Backfill title/preview for existing sessions |

#### Create Session Request

```bash
curl -X POST http://localhost:8080/api/chat/sessions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbG..." \
  -d '{
    "message": "Help me refactor the auth module",
    "cwd": "/path/to/project",
    "project_slug": "my-project",
    "model": "claude-opus-4-6"
  }'
```

#### List Messages

```bash
curl "http://localhost:8080/api/chat/sessions/SESSION_UUID/messages?limit=50&offset=0" \
  -H "Authorization: Bearer eyJhbG..."
```

#### Search Messages

```bash
curl "http://localhost:8080/api/chat/search?q=authentication&project_slug=my-api&limit=10" \
  -H "Authorization: Bearer eyJhbG..."
```

### Session Lifecycle

1. **Create** -- `POST /api/chat/sessions` creates a new session in Neo4j and starts a Claude CLI subprocess
2. **Connect** -- Client connects to `/ws/chat/{session_id}` for real-time streaming
3. **Chat** -- Send messages via WebSocket, receive streaming events
4. **Idle timeout** -- After `session_timeout_secs` (default: 1800 = 30 minutes) of inactivity, the CLI subprocess is freed
5. **Resume** -- Sending a message to an inactive session automatically resumes the CLI process
6. **Delete** -- `DELETE /api/chat/sessions/{id}` closes the subprocess and removes the session from Neo4j

### Session Data Model

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "cli_session_id": "cli-abc123",
  "project_slug": "my-project",
  "cwd": "/path/to/project",
  "title": "Refactoring auth module",
  "model": "claude-opus-4-6",
  "created_at": "2026-02-10T10:00:00Z",
  "updated_at": "2026-02-10T10:15:00Z",
  "message_count": 12,
  "total_cost_usd": 0.45,
  "conversation_id": "conv-xyz-789",
  "preview": "Help me refactor the auth module"
}
```

---

## MCP Chat Tools (5)

These tools are available via the MCP protocol for programmatic access:

| Tool | Description |
|------|-------------|
| `list_chat_sessions` | List sessions with optional project filter and pagination |
| `get_chat_session` | Get session details by ID |
| `delete_chat_session` | Delete a session |
| `list_chat_messages` | List message history for a session (chronological order) |
| `chat_send_message` | Send a message and wait for the complete response (non-streaming, blocks until Claude finishes) |

### `chat_send_message`

This tool is designed for MCP clients that cannot handle streaming. It sends a message and waits for the complete response:

```json
{
  "message": "Search for authentication code",
  "cwd": "/path/to/project",
  "project_slug": "my-project",
  "session_id": "optional-existing-session-uuid",
  "model": "claude-opus-4-6"
}
```

---

## Configuration

The chat system is configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHAT_DEFAULT_MODEL` | Default Claude model | `claude-opus-4-6` |
| `CHAT_MAX_SESSIONS` | Maximum concurrent active sessions | `10` |
| `CHAT_SESSION_TIMEOUT_SECS` | Idle timeout before subprocess is freed | `1800` (30 min) |
| `CHAT_MAX_TURNS` | Maximum agentic turns (tool calls) per message | `50` |
| `PROMPT_BUILDER_MODEL` | Model for oneshot prompt builder (context refinement) | `claude-opus-4-6` |
| `MCP_SERVER_PATH` | Path to the MCP server binary | Auto-detected |

The chat system also inherits Neo4j and Meilisearch connection settings from the main configuration (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `MEILISEARCH_URL`, `MEILISEARCH_KEY`).

### NATS Configuration (Multi-Instance)

When NATS is configured, the chat system supports cross-instance routing:

| Variable | Description | Default |
|----------|-------------|---------|
| `NATS_URL` | NATS server URL (e.g., `nats://localhost:4222`) | _(none)_ |

With NATS enabled:
- **Chat messages** are routed to the instance hosting the active CLI session via request/reply RPC
- **Interrupt signals** are forwarded across instances
- **Streaming snapshots** are shared for mid-stream joins from other instances
- **CRUD events** are published to NATS and broadcast locally via HybridEmitter (with deduplication)

Without NATS, the orchestrator operates in single-instance mode with local broadcasting only.

---

## Troubleshooting

### WebSocket won't connect

- **Auth timeout**: The server waits 10 seconds for the auth message. Ensure you send `{"type": "auth", "token": "..."}` immediately after `onopen`.
- **Invalid token**: Check that the JWT is valid and not expired. The server sends `auth_error` with details.
- **Email domain restriction**: If `allowed_email_domain` is configured, the JWT email must match.

### Session not found (404)

The session must exist in Neo4j before connecting to the WebSocket. Create it first via `POST /api/chat/sessions` or the `chat_send_message` MCP tool.

### Session expired / inactive

When a session's CLI subprocess times out, it is freed but the session data remains. Sending a new `user_message` via WebSocket automatically resumes the session. No action needed.

### No events on `/ws/events`

- Verify the WebSocket is authenticated (`auth_ok` received).
- Check your `entity_types` filter -- if specified, only matching events are forwarded.
- Ensure mutations are happening through the API/MCP layer (direct Neo4j changes are not detected).

### Events lagged

If the client is too slow to consume events, the broadcast channel drops older events. The server sends `{"type": "events_lagged", "skipped": N}`. The client may want to do a full state refresh from the REST API.

### Chat manager not initialized

If REST endpoints return "Chat manager not initialized", the server was started without the required chat configuration (missing `MCP_SERVER_PATH` or nexus-claude SDK not available).

---

## JavaScript Client Example

```javascript
class ChatClient {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  connect(sessionId, onEvent) {
    const ws = new WebSocket(`${this.baseUrl}/ws/chat/${sessionId}`);

    ws.onopen = () => {
      // Step 1: Authenticate
      ws.send(JSON.stringify({ type: "auth", token: this.token }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "auth_ok":
          console.log("Authenticated as", data.user.email);
          break;
        case "replay_complete":
          console.log("Replay finished, ready for live events");
          break;
        case "assistant_text":
          onEvent("text", data.content);
          break;
        case "tool_use":
          onEvent("tool", { id: data.id, tool: data.tool, input: data.input });
          break;
        case "tool_result":
          onEvent("tool_result", { id: data.id, result: data.result });
          break;
        case "result":
          onEvent("done", { duration: data.duration_ms, cost: data.cost_usd });
          break;
        case "error":
          onEvent("error", data.message);
          break;
      }
    };

    return {
      send: (content) => ws.send(JSON.stringify({ type: "user_message", content })),
      interrupt: () => ws.send(JSON.stringify({ type: "interrupt" })),
      close: () => ws.close(),
    };
  }
}
```

---

## API Reference

See [MCP Tools Reference](../api/mcp-tools.md) for complete MCP tool documentation.

See [API Reference](../api/reference.md) for complete REST endpoint documentation.
