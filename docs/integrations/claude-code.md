# Claude Code Integration

Complete guide to integrating Project Orchestrator with Claude Code (Anthropic's CLI for Claude).

---

## Overview

The integration gives Claude Code access to **19 mega-tools** (each with multiple actions) for:

- **Project Management** — Create, sync, and explore codebases
- **Plan & Task Tracking** — Manage development workflows with dependencies
- **Code Intelligence** — Semantic search, impact analysis, call graphs
- **Decision Recording** — Track architectural decisions across sessions
- **Knowledge Notes** — Persistent notes anchored to code entities
- **Chat** — Delegate tasks to sub-agents via WebSocket or MCP
- **Authentication** — OAuth (Google) and API key support for the HTTP API

---

## Configuration

### Step 1: Locate your MCP configuration

Claude Code stores MCP server configurations in:

```
~/.claude/mcp.json
```

Create the file if it doesn't exist:

```bash
mkdir -p ~/.claude
touch ~/.claude/mcp.json
```

### Step 2: Add Project Orchestrator

Add the following to your `mcp.json`:

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "orchestrator123",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me"
      }
    }
  }
}
```

**Important:** Replace `/path/to/mcp_server` with the absolute path to your binary.

### Alternative: Using CLI arguments

You can also pass configuration via arguments:

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
      "args": [
        "--neo4j-uri", "bolt://localhost:7687",
        "--neo4j-user", "neo4j",
        "--neo4j-password", "orchestrator123",
        "--meilisearch-url", "http://localhost:7700",
        "--meilisearch-key", "orchestrator-meili-key-change-me"
      ]
    }
  }
}
```

### Alternative: Auto-Configure

If you have the `orchestrator` binary installed, you can auto-configure Claude Code:

```bash
orchestrator setup-claude
```

This detects your Claude Code installation and adds the MCP server to `~/.claude/mcp.json` automatically.

### Step 3: Restart Claude Code

After modifying `mcp.json`, restart Claude Code to load the new configuration:

```bash
claude --mcp-restart
# or simply restart the terminal
```

---

## Verification

### Check MCP is loaded

In Claude Code, run:

```
/mcp
```

You should see `project-orchestrator` in the list of connected servers.

### Test a tool

Ask Claude:

```
List all registered projects
```

Claude should use the `project` tool with action `list` and return results.

---

## Available Mega-Tools (19)

Project Orchestrator uses a **mega-tool architecture**: 19 tools, each with an `action` parameter that dispatches to specific operations. This keeps the tool count low while providing comprehensive functionality.

### How it works

```json
// Example: List all projects
{ "tool": "project", "action": "list" }

// Example: Search code in a project
{ "tool": "code", "action": "search_project", "slug": "my-project", "query": "authentication" }
```

### Quick Reference

| Mega-Tool | Actions | Description |
|-----------|---------|-------------|
| `project` | 8 | Project CRUD, sync, roadmap |
| `plan` | 10 | Plans, status, dependencies, critical path |
| `task` | 13 | Tasks, dependencies, blockers, context, prompt |
| `step` | 6 | Sub-steps within tasks |
| `decision` | 12 | Architectural decisions, semantic search, affects |
| `constraint` | 5 | Plan constraints (performance, security, style) |
| `release` | 8 | Release management with tasks and commits |
| `milestone` | 9 | Milestones with progress tracking |
| `commit` | 7 | Git commit tracking, file history |
| `note` | 20 | Knowledge notes, semantic search, propagation |
| `workspace` | 10 | Multi-project workspaces, topology |
| `workspace_milestone` | 10 | Cross-project milestones |
| `resource` | 6 | API contracts, schemas, shared resources |
| `component` | 8 | Service architecture, dependencies |
| `chat` | 7 | Chat sessions, sub-agent delegation |
| `feature_graph` | 6 | Feature dependency graphs |
| `code` | 30 | Code search, impact analysis, call graphs, health metrics |
| `admin` | 23 | Sync, watch, Knowledge Fabric, neural maintenance |
| `skill` | 12 | Neural skills: emergent knowledge clusters |

For detailed documentation of every action and parameter, see the [MCP Tools Reference](../api/mcp-tools.md).

---

## Workflows

### Workflow 1: Starting a New Project

```
You: Register my project at /Users/me/myapp

Claude: [Uses project(action: "create")]
        Created project "myapp" with slug "myapp"

You: Now sync it so you can understand the code

Claude: [Uses project(action: "sync")]
        Synced 342 files. Found 128 functions, 45 structs, 12 traits.

You: Show me the architecture overview

Claude: [Uses code(action: "get_architecture")]
        Most connected files:
        - src/lib.rs (imported by 23 files)
        - src/models/user.rs (imported by 15 files)
        ...
```

### Workflow 2: Working on a Plan

```
You: What's the next task I should work on for plan abc123?

Claude: [Uses task(action: "get_next")]
        Next task: "Implement user authentication"
        Priority: 10
        Tags: [backend, security]

You: Okay, I'm starting on it

Claude: [Uses task(action: "update", status: "in_progress")]
        Task marked as in progress.

You: I decided to use JWT instead of sessions. Record that.

Claude: [Uses decision(action: "add")]
        Decision recorded:
        - Description: Use JWT instead of sessions
        - Rationale: Better for stateless API

You: Done! Mark it complete.

Claude: [Uses task(action: "update", status: "completed")]
        Task completed! Next available task is...
```

### Workflow 3: Code Exploration

```
You: Find all code related to error handling

Claude: [Uses code(action: "search")]
        Found 15 results:
        1. src/api/handlers.rs - AppError struct
        2. src/lib.rs - Error handling middleware
        ...

You: What would be impacted if I change the AppError struct?

Claude: [Uses code(action: "analyze_impact")]
        Impact analysis for AppError:
        - Directly affected: 8 files
        - Transitively affected: 23 files
        - Test files affected: 5
        - Risk level: High

You: Show me everything that calls the handle_request function

Claude: [Uses code(action: "get_call_graph")]
        Call graph for handle_request:
        Callers: main, route_handler, test_handler
        Callees: validate_input, process_request, send_response
```

### Workflow 4: Multi-Agent Coordination

```
Agent 1: Get the next task for plan xyz

Claude: [Uses task(action: "get_next")]
        Task: "Implement login endpoint"

Agent 1: [Works on task, completes it]

Agent 1: Mark task as completed and record my commit

Claude: [Uses task(action: "update"), commit(action: "create"), commit(action: "link_to_task")]
        Task completed. Commit abc123 linked.

Agent 2: Get the next task (runs in parallel)

Claude: [Uses task(action: "get_next")]
        Task: "Implement logout endpoint" (was blocked by login, now available)
```

### Workflow 5: Authentication Setup

If your deployment uses the HTTP API (not just MCP via stdio), you can set up authentication:

```
You: Set up Google OAuth for the orchestrator

Claude: The Project Orchestrator HTTP API supports two authentication methods:

        1. **Google OAuth** — For browser-based access (dashboard, WebSocket chat)
        2. **API Keys** — For programmatic access (CI/CD, scripts)

        To enable Google OAuth:

        1. Create a Google Cloud project and OAuth credentials
        2. Set the following environment variables:
           - GOOGLE_CLIENT_ID=your-client-id
           - GOOGLE_CLIENT_SECRET=your-client-secret
           - JWT_SECRET=your-jwt-secret
           - FRONTEND_URL=http://localhost:3000

        3. Restart the server. The auth endpoints will be available:
           - GET /auth/google — Start OAuth flow
           - GET /auth/callback — OAuth callback
           - GET /auth/me — Get current user

        See the Authentication Guide for full details.
```

> **Note:** MCP tools (via stdio) do not require authentication — they run locally as a child process. Authentication is only needed when accessing the HTTP REST API or WebSocket chat endpoint.

### Workflow 6: Delegating Work via Chat

```
You: Send a task to a sub-agent to analyze the auth module

Claude: [Uses chat(action: "send_message")]
        Sent message: "Analyze the auth module structure and report key findings"
        Working directory: /Users/me/myapp

        Response from sub-agent:
        The auth module has 3 main files:
        - auth/mod.rs - Module entry point
        - auth/jwt.rs - JWT token handling
        - auth/middleware.rs - Axum auth middleware
        ...

You: List all my chat sessions

Claude: [Uses chat(action: "list_sessions")]
        1. Session abc-123 (project: myapp) — 5 messages
        2. Session def-456 (project: myapp) — 12 messages
```

---

## Tips & Best Practices

### 1. Sync before exploring

Always sync your project before asking code-related questions:

```
Sync the project, then show me functions related to authentication
```

### 2. Use semantic search

Instead of asking for specific file paths, describe what you're looking for:

```
Find code that handles database connections
```

### 3. Record decisions as you go

When you make architectural choices, record them immediately:

```
Record a decision: we're using Redis for caching because it supports pub/sub
```

### 4. Check impact before refactoring

Before major changes, check what might break:

```
What would be impacted if I rename the UserService class?
```

### 5. Use milestones for long-term planning

Group related tasks into milestones for better tracking:

```
Create a milestone "v1.0 Release" with target date March 1st
```

---

## Debug & Troubleshooting

### Enable debug logging

Add `RUST_LOG=debug` to your MCP configuration:

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
      "env": {
        "RUST_LOG": "debug",
        "NEO4J_URI": "bolt://localhost:7687",
        ...
      }
    }
  }
}
```

Logs are written to stderr (stdout is reserved for MCP protocol).

### Common issues

**"Connection refused" errors**

Ensure the backend services are running:

```bash
docker compose ps
docker compose up -d neo4j meilisearch
```

**"Tool not found" errors**

Restart Claude Code to reload MCP configuration:

```bash
claude --mcp-restart
```

**MCP server crashes on startup**

Check the binary path is correct and executable:

```bash
ls -la /path/to/mcp_server
chmod +x /path/to/mcp_server
```

### View MCP logs

MCP server logs go to stderr. To capture them:

```bash
/path/to/mcp_server 2>/tmp/mcp.log
```

---

## Example mcp.json (Complete)

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/Users/me/.local/bin/mcp_server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "orchestrator123",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me",
        "NATS_URL": "nats://localhost:4222",
        "RUST_LOG": "info"
      }
    }
  }
}
```

---

## Next Steps

- [Getting Started Tutorial](../guides/getting-started.md) — Full walkthrough
- [Authentication Guide](../guides/authentication.md) — OAuth, API keys, and JWT setup
- [Chat & WebSocket Guide](../guides/chat-websocket.md) — Sub-agent delegation and real-time chat
- [Knowledge Notes Guide](../guides/knowledge-notes.md) — Persistent notes anchored to code
- [API Reference](../api/reference.md) — REST API documentation
- [MCP Tools Reference](../api/mcp-tools.md) — Detailed tool documentation
- [Multi-Agent Workflows](../guides/multi-agent-workflow.md) — Advanced coordination
