# AGENTS.md

Guidelines for AI agents working with Project Orchestrator.

> **Detailed guides available:**
> - [Getting Started Tutorial](docs/guides/getting-started.md)
> - [Multi-Agent Workflow Guide](docs/guides/multi-agent-workflow.md)
> - [MCP Tools Reference](docs/api/mcp-tools.md)

---

## Quick Start Workflow

### 1. Get your task

```bash
# Get the next available task (unblocked, highest priority)
GET /api/plans/{plan_id}/next-task
```

### 2. Claim it

```bash
PATCH /api/tasks/{task_id}
{"status": "in_progress", "assigned_to": "agent-id"}
```

### 3. Work through steps

For each step in the task:
1. Read the description
2. Do the work
3. Mark step completed: `PATCH /api/steps/{step_id} {"status": "completed"}`

### 4. Record decisions

When you make important choices:

```bash
POST /api/tasks/{task_id}/decisions
{
  "description": "Use JWT instead of sessions",
  "rationale": "Better for stateless API",
  "alternatives": ["Sessions", "OAuth"],
  "chosen_option": "JWT with refresh tokens"
}
```

### 5. Commit your changes

Format:
```
<type>(<scope>): <description>

<Task title>

Changes:
- <change 1>
- <change 2>

Co-Authored-By: <Agent Name> <email>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### 6. Mark complete

```bash
PATCH /api/tasks/{task_id}
{"status": "completed"}
```

---

## Checklist Before Completion

- [ ] All steps completed
- [ ] Tests pass
- [ ] Constraints respected (clippy, tests, etc.)
- [ ] Changes committed
- [ ] Task marked as `completed`

---

## Getting Help

### Search for code patterns
```bash
GET /api/code/search?q=error+handling&limit=10
```

### Check impact before changes
```bash
GET /api/code/impact?target=src/models/user.rs
```

### Find past decisions
```bash
GET /api/decisions/search?q=authentication
```

---

## If Something Goes Wrong

**Do NOT mark task as `completed`.**

Instead:
1. Mark as `failed` with explanation
2. Or mark as `blocked` if waiting on something

```bash
PATCH /api/tasks/{task_id}
{"status": "failed"}

POST /api/wake
{
  "task_id": "uuid",
  "success": false,
  "summary": "Tests failing after X modification"
}
```

---

## Authentication

If auth is enabled on the server, agents using the REST API need a JWT token:

```bash
# Get token via Google OAuth flow, then use it:
curl -H "Authorization: Bearer <JWT_TOKEN>" http://localhost:8080/api/plans
```

**Note:** MCP server (stdio) does not require auth. WebSocket connections authenticate via first message.

See [Authentication Guide](docs/guides/authentication.md) for details.

---

## Full Documentation

- [Getting Started](docs/guides/getting-started.md) — Complete tutorial
- [Authentication](docs/guides/authentication.md) — JWT + Google OAuth setup
- [Chat & WebSocket](docs/guides/chat-websocket.md) — Real-time chat and events
- [Multi-Agent Workflows](docs/guides/multi-agent-workflow.md) — Coordinating multiple agents
- [API Reference](docs/api/reference.md) — All REST endpoints
- [MCP Tools](docs/api/mcp-tools.md) — 137 tools documented
