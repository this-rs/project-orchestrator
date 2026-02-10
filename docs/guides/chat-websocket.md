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
                                |
                           Neo4j (sessions, messages, events)
```

- **ChatManager** manages active sessions, broadcast channels, and the nexus-claude SDK integration
- **Neo4j** persists session metadata, message history, and structured chat events for replay
- **Meilisearch** indexes messages for full-text search across sessions
- **EventBus** broadcasts CRUD events to all connected `/ws/events` clients

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

#### `permission_response` -- Respond to a permission request

```json
{"type": "permission_response", "id": "pr_1", "allow": true}
```

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
| `input_request` | Claude needs user input | `prompt`, `options` (optional array) |
| `result` | Conversation turn completed | `session_id`, `duration_ms`, `cost_usd` (optional) |
| `stream_delta` | Raw streaming text token (real-time) | `text` |
| `streaming_status` | Stream state change | `is_streaming` (boolean) |
| `error` | An error occurred | `message` |

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
