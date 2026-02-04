# Claude Code Integration

Complete guide to integrating Project Orchestrator with Claude Code (Anthropic's CLI for Claude).

---

## Overview

The integration gives Claude Code access to **62 MCP tools** for:

- **Project Management** — Create, sync, and explore codebases
- **Plan & Task Tracking** — Manage development workflows with dependencies
- **Code Intelligence** — Semantic search, impact analysis, call graphs
- **Decision Recording** — Track architectural decisions across sessions

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

Claude should use the `list_projects` tool and return results.

---

## Available Tools (62)

### Project Management (6 tools)

| Tool | Description |
|------|-------------|
| `list_projects` | List all projects with optional search and pagination |
| `create_project` | Create a new project to track a codebase |
| `get_project` | Get project details by slug |
| `delete_project` | Delete a project and all associated data |
| `sync_project` | Sync a project's codebase (parse files, update graph) |
| `get_project_roadmap` | Get aggregated roadmap view with milestones and releases |

### Plan Management (8 tools)

| Tool | Description |
|------|-------------|
| `list_plans` | List plans with optional filters and pagination |
| `create_plan` | Create a new development plan |
| `get_plan` | Get plan details including tasks, constraints, and decisions |
| `update_plan_status` | Update a plan's status |
| `link_plan_to_project` | Link a plan to a project |
| `unlink_plan_from_project` | Unlink a plan from its project |
| `get_dependency_graph` | Get the task dependency graph for a plan |
| `get_critical_path` | Get the critical path (longest dependency chain) |

### Task Management (12 tools)

| Tool | Description |
|------|-------------|
| `list_tasks` | List all tasks across plans with filters |
| `create_task` | Add a new task to a plan |
| `get_task` | Get task details including steps and decisions |
| `update_task` | Update a task's status, assignee, or other fields |
| `get_next_task` | Get the next available task (unblocked, highest priority) |
| `add_task_dependencies` | Add dependencies to a task |
| `remove_task_dependency` | Remove a dependency from a task |
| `get_task_blockers` | Get tasks that are blocking this task |
| `get_tasks_blocked_by` | Get tasks that are blocked by this task |
| `get_task_context` | Get full context for a task (for agent execution) |
| `get_task_prompt` | Get generated prompt for a task |
| `add_decision` | Record an architectural decision for a task |

### Step Management (4 tools)

| Tool | Description |
|------|-------------|
| `list_steps` | List all steps for a task |
| `create_step` | Add a step to a task |
| `update_step` | Update a step's status |
| `get_step_progress` | Get step completion progress for a task |

### Constraint Management (3 tools)

| Tool | Description |
|------|-------------|
| `list_constraints` | List constraints for a plan |
| `add_constraint` | Add a constraint to a plan |
| `delete_constraint` | Delete a constraint |

### Release Management (5 tools)

| Tool | Description |
|------|-------------|
| `list_releases` | List releases for a project |
| `create_release` | Create a new release for a project |
| `get_release` | Get release details with tasks and commits |
| `update_release` | Update a release |
| `add_task_to_release` | Add a task to a release |

### Milestone Management (5 tools)

| Tool | Description |
|------|-------------|
| `list_milestones` | List milestones for a project |
| `create_milestone` | Create a new milestone for a project |
| `get_milestone` | Get milestone details with tasks |
| `update_milestone` | Update a milestone |
| `get_milestone_progress` | Get milestone completion progress |

### Commit Tracking (4 tools)

| Tool | Description |
|------|-------------|
| `create_commit` | Register a git commit |
| `link_commit_to_task` | Link a commit to a task (RESOLVED_BY) |
| `link_commit_to_plan` | Link a commit to a plan (RESULTED_IN) |
| `get_task_commits` | Get commits linked to a task |

### Code Exploration (10 tools)

| Tool | Description |
|------|-------------|
| `search_code` | Search code semantically across all projects |
| `search_project_code` | Search code within a specific project |
| `get_file_symbols` | Get all symbols in a file (functions, structs, traits) |
| `find_references` | Find all references to a symbol |
| `get_file_dependencies` | Get file imports and files that depend on it |
| `get_call_graph` | Get the call graph for a function |
| `analyze_impact` | Analyze the impact of changing a file or symbol |
| `get_architecture` | Get codebase architecture overview |
| `find_similar_code` | Find code similar to a given snippet |
| `find_trait_implementations` | Find all implementations of a trait |

### Decision Search (1 tool)

| Tool | Description |
|------|-------------|
| `search_decisions` | Search architectural decisions |

### Sync & Watch (4 tools)

| Tool | Description |
|------|-------------|
| `sync_directory` | Manually sync a directory to the knowledge graph |
| `start_watch` | Start auto-sync file watcher |
| `stop_watch` | Stop the file watcher |
| `watch_status` | Get file watcher status |

---

## Workflows

### Workflow 1: Starting a New Project

```
You: Register my project at /Users/me/myapp

Claude: [Uses create_project tool]
        Created project "myapp" with slug "myapp"

You: Now sync it so you can understand the code

Claude: [Uses sync_project tool]
        Synced 342 files. Found 128 functions, 45 structs, 12 traits.

You: Show me the architecture overview

Claude: [Uses get_architecture tool]
        Most connected files:
        - src/lib.rs (imported by 23 files)
        - src/models/user.rs (imported by 15 files)
        ...
```

### Workflow 2: Working on a Plan

```
You: What's the next task I should work on for plan abc123?

Claude: [Uses get_next_task tool]
        Next task: "Implement user authentication"
        Priority: 10
        Tags: [backend, security]

You: Okay, I'm starting on it

Claude: [Uses update_task tool with status: "in_progress"]
        Task marked as in progress.

You: I decided to use JWT instead of sessions. Record that.

Claude: [Uses add_decision tool]
        Decision recorded:
        - Description: Use JWT instead of sessions
        - Rationale: Better for stateless API

You: Done! Mark it complete.

Claude: [Uses update_task tool with status: "completed"]
        Task completed! Next available task is...
```

### Workflow 3: Code Exploration

```
You: Find all code related to error handling

Claude: [Uses search_code tool]
        Found 15 results:
        1. src/api/handlers.rs - AppError struct
        2. src/lib.rs - Error handling middleware
        ...

You: What would be impacted if I change the AppError struct?

Claude: [Uses analyze_impact tool]
        Impact analysis for AppError:
        - Directly affected: 8 files
        - Transitively affected: 23 files
        - Test files affected: 5
        - Risk level: High

You: Show me everything that calls the handle_request function

Claude: [Uses get_call_graph tool]
        Call graph for handle_request:
        Callers: main, route_handler, test_handler
        Callees: validate_input, process_request, send_response
```

### Workflow 4: Multi-Agent Coordination

```
Agent 1: Get the next task for plan xyz

Claude: [Uses get_next_task]
        Task: "Implement login endpoint"

Agent 1: [Works on task, completes it]

Agent 1: Mark task as completed and record my commit

Claude: [Uses update_task, create_commit, link_commit_to_task]
        Task completed. Commit abc123 linked.

Agent 2: Get the next task (runs in parallel)

Claude: [Uses get_next_task]
        Task: "Implement logout endpoint" (was blocked by login, now available)
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
        "RUST_LOG": "info"
      }
    }
  }
}
```

---

## Next Steps

- [Getting Started Tutorial](../guides/getting-started.md) — Full walkthrough
- [API Reference](../api/reference.md) — REST API documentation
- [MCP Tools Reference](../api/mcp-tools.md) — Detailed tool documentation
- [Multi-Agent Workflows](../guides/multi-agent-workflow.md) — Advanced coordination
