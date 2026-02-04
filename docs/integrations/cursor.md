# Cursor Integration

Guide to integrating Project Orchestrator with Cursor IDE.

---

## Overview

Cursor supports MCP (Model Context Protocol) servers, giving you access to Project Orchestrator's **62 tools** directly in the editor. Use Chat or Composer to:

- **Explore code** with semantic search and graph queries
- **Manage plans and tasks** without leaving your editor
- **Track decisions** as you work
- **Analyze impact** before refactoring

---

## Prerequisites

1. **Cursor IDE** installed (version 0.40+ with MCP support)
2. **Backend services running** (see [Installation Guide](../setup/installation.md))
3. **MCP server binary built**

```bash
# Build the MCP server
cargo build --release --bin mcp_server

# Verify it works
./target/release/mcp_server --help
```

---

## Configuration

### Step 1: Open Cursor Settings

1. Open Cursor
2. Press `Cmd+,` (macOS) or `Ctrl+,` (Windows/Linux)
3. Search for "MCP" in settings
4. Click "Edit in settings.json"

### Step 2: Add MCP Server Configuration

Add the following to your Cursor `settings.json`:

```json
{
  "mcp.servers": {
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

### Step 3: Restart Cursor

After saving the configuration, restart Cursor to load the MCP server.

---

## Alternative: Project-specific Configuration

For project-specific settings, create `.cursor/mcp.json` in your project root:

```json
{
  "servers": {
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

This configuration will only apply when working in that project.

---

## Verification

### Check MCP Status

1. Open Cursor Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Search for "MCP: Show Servers"
3. Verify `project-orchestrator` is listed and connected

### Test in Chat

Open Cursor Chat (`Cmd+L` / `Ctrl+L`) and ask:

```
List all registered projects in the orchestrator
```

Cursor should use the `list_projects` tool and return results.

---

## Using with Cursor Chat

### Code Exploration

```
User: Find all functions that handle authentication

Cursor: [Uses search_code tool]
        Found 8 results related to authentication:
        1. src/auth/login.rs:45 - fn authenticate_user()
        2. src/auth/jwt.rs:23 - fn validate_token()
        ...
```

### Impact Analysis

```
User: What would be affected if I change the UserService struct?

Cursor: [Uses analyze_impact tool]
        Impact analysis:
        - 12 files directly import UserService
        - 28 files transitively affected
        - 3 test files would need updates
        - Risk level: Medium
```

### Task Management

```
User: What's my next task on the authentication plan?

Cursor: [Uses get_next_task tool]
        Next available task:
        - Title: Implement password reset
        - Priority: 8
        - Status: pending
        - Dependencies: None (ready to start)
```

---

## Using with Composer

Composer can use MCP tools for multi-step operations:

### Example: Plan a Feature

```
User: Create a plan to add OAuth support to this project

Cursor Composer:
1. [Uses create_plan tool]
   Created plan "OAuth Integration" (id: abc-123)

2. [Uses create_task tool]
   Added task: "Research OAuth providers"

3. [Uses create_task tool]
   Added task: "Implement OAuth flow"

4. [Uses add_task_dependencies tool]
   Set implementation depends on research

Done! Created plan with 2 tasks.
```

### Example: Refactor with Context

```
User: Refactor the error handling in this project

Cursor Composer:
1. [Uses search_code tool]
   Found error handling in 15 files

2. [Uses analyze_impact tool]
   Analyzing impact of changes...

3. [Uses add_decision tool]
   Recording decision: "Standardize on thiserror crate"

4. [Makes code changes with full context]
```

---

## Workflows

### Workflow 1: Start Your Day

```
"Show me my in-progress tasks and what's blocked"
```

Cursor will use `list_tasks` to show your current work status.

### Workflow 2: Before Refactoring

```
"I want to rename AuthService to AuthenticationService.
What files would be affected?"
```

Cursor will use `analyze_impact` and `find_references` to show the scope.

### Workflow 3: Code Review Context

```
"What decisions have been made about authentication in this project?"
```

Cursor will use `search_decisions` to find past architectural choices.

### Workflow 4: End of Day

```
"Mark task xyz as completed and summarize what I did today"
```

Cursor will update the task and can generate a summary from your commits.

---

## Tips for Cursor

### 1. Use @-mentions with MCP

Combine Cursor's file context with MCP tools:

```
@src/auth/login.rs What functions in this file call external services?
Show me the call graph.
```

### 2. Inline MCP in Composer

When using Composer for code changes, ask for context first:

```
Before making changes, search for similar patterns in the codebase,
then refactor this function to match the project style.
```

### 3. Project Sync on Open

Add to your workflow: sync the project when you open Cursor:

```
Sync the project-orchestrator project to refresh the code index
```

### 4. Decision Recording

Record decisions as you make them:

```
I decided to use async/await here instead of callbacks.
Record this decision with rationale: better error handling and readability.
```

---

## Available Tools

All 62 Project Orchestrator tools are available in Cursor. Most useful for editor workflows:

| Tool | Cursor Use Case |
|------|-----------------|
| `search_code` | Find code patterns before editing |
| `get_file_symbols` | Understand file structure |
| `analyze_impact` | Check scope before refactoring |
| `find_references` | Find all usages of a symbol |
| `get_call_graph` | Understand function relationships |
| `list_tasks` | See your work queue |
| `update_task` | Mark progress |
| `add_decision` | Record architectural choices |
| `search_decisions` | Review past decisions |

See [Claude Code Integration](./claude-code.md#available-tools-62) for the complete tool list.

---

## Differences from Claude Code

| Feature | Claude Code | Cursor |
|---------|-------------|--------|
| **Config location** | `~/.claude/mcp.json` | `settings.json` or `.cursor/mcp.json` |
| **Interface** | Terminal | GUI (Chat + Composer) |
| **File context** | Manual | Automatic with @-mentions |
| **Multi-file edits** | Sequential | Composer handles multiple |
| **Git integration** | CLI commands | Built-in UI |

### Cursor-specific advantages

1. **Visual file context**: @-mention files for automatic context
2. **Composer**: Multi-file edits with MCP tool access
3. **Inline suggestions**: MCP can inform autocomplete context
4. **Project awareness**: Cursor knows your open project

---

## Troubleshooting

### MCP server not connecting

1. Check the binary path is absolute and correct
2. Verify the binary is executable: `chmod +x /path/to/mcp_server`
3. Restart Cursor after config changes

### "Tool not found" errors

1. Open Command Palette → "MCP: Refresh Servers"
2. Check MCP server logs (see below)
3. Restart Cursor

### View MCP logs

Enable debug logging in your configuration:

```json
{
  "mcp.servers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
      "env": {
        "RUST_LOG": "debug",
        ...
      }
    }
  }
}
```

View logs in Cursor:
1. Open Command Palette
2. Search "MCP: Show Server Logs"
3. Select `project-orchestrator`

### Backend connection issues

```bash
# Verify services are running
docker compose ps

# Check health endpoints
curl http://localhost:8080/health
curl http://localhost:7700/health
```

---

## Example settings.json

Complete configuration for Cursor:

```json
{
  "mcp.servers": {
    "project-orchestrator": {
      "command": "/Users/me/tools/mcp_server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "orchestrator123",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "orchestrator-meili-key-change-me",
        "RUST_LOG": "info"
      }
    }
  },
  "mcp.enableAutoConnect": true,
  "mcp.showToolCalls": true
}
```

---

## Next Steps

- [Getting Started Tutorial](../guides/getting-started.md)
- [API Reference](../api/reference.md)
- [MCP Tools Reference](../api/mcp-tools.md)
- [Multi-Agent Workflows](../guides/multi-agent-workflow.md)
