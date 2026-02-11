# Getting Started

A step-by-step tutorial to get Project Orchestrator running with your first project.

**Time:** ~15 minutes

---

## What You'll Learn

1. Start the backend services
2. Configure your AI tool (Claude Code, OpenAI, or Cursor)
3. Register and sync a project
4. Explore code with semantic search
5. Create a plan with tasks
6. Work through a task
7. Record decisions
8. Track progress

---

## Prerequisites

- Docker and Docker Compose installed
- An AI tool: Claude Code, OpenAI Agents SDK, or Cursor
- A codebase to analyze (we'll use a sample project)

---

## Step 1: Start Backend Services

Clone and start the orchestrator:

```bash
# Clone the repository
git clone https://github.com/this-rs/project-orchestrator.git
cd project-orchestrator

# Start Neo4j and Meilisearch
docker compose up -d neo4j meilisearch

# Wait for services to be healthy (about 30 seconds)
docker compose ps
```

You should see both services as "healthy":

```
NAME                    STATUS
orchestrator-neo4j      running (healthy)
orchestrator-meilisearch running (healthy)
```

### Build the MCP server

```bash
cargo build --release --bin mcp_server
```

The binary is at `./target/release/mcp_server`.

**Alternative:** Install the pre-built binary:

```bash
curl -fsSL https://raw.githubusercontent.com/this-rs/project-orchestrator/main/install.sh | sh
```

Then auto-configure Claude Code:

```bash
orchestrator setup-claude
```

---

## Step 2: Configure Your AI Tool

Choose your tool and follow the configuration:

### Option A: Claude Code

Create or edit `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/full/path/to/mcp_server",
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

Restart Claude Code to load the configuration.

### Option B: Cursor

Add to your Cursor `settings.json`:

```json
{
  "mcp.servers": {
    "project-orchestrator": {
      "command": "/full/path/to/mcp_server",
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

Restart Cursor.

### Option C: OpenAI Agents

See [OpenAI Integration Guide](../integrations/openai.md) for code-based configuration.

---

## Step 3: Verify the Connection

Ask your AI tool:

```
List all projects in the orchestrator
```

Expected response:

```
No projects found. The orchestrator is ready to register projects.
```

If you get an error, check:
- Backend services are running: `docker compose ps`
- MCP server path is correct and absolute
- You restarted your AI tool after configuration

---

## Step 4: Register Your First Project

Let's register a project. You can use any codebase you have locally.

Ask your AI tool:

```
Create a new project called "my-app" at /path/to/your/codebase
```

The AI will use the `create_project` tool:

```
Created project:
- Name: my-app
- Slug: my-app
- Path: /path/to/your/codebase
```

---

## Step 5: Sync the Codebase

Now let's parse and index your code:

```
Sync the my-app project
```

The AI will use `sync_project`:

```
Synced project my-app:
- Files parsed: 127
- Functions found: 89
- Structs/Classes: 34
- Imports tracked: 256
```

This creates a knowledge graph of your code structure.

---

## Step 6: Explore Your Code

Now the fun part! Try these queries:

### Search for code

```
Search for code related to "authentication" in my-app
```

Result:
```
Found 5 matches:
1. src/auth/login.rs - authenticate_user function
2. src/middleware/auth.rs - AuthMiddleware struct
3. src/routes/auth.rs - login_handler function
...
```

### View file structure

```
Show me all symbols in src/lib.rs
```

Result:
```
Symbols in src/lib.rs:
- Functions: main, setup_routes, configure_app
- Structs: AppState, Config
- Imports: tokio, axum, serde
```

### Check dependencies

```
What files import src/models/user.rs?
```

Result:
```
Files that import user.rs:
- src/handlers/user_handler.rs
- src/services/user_service.rs
- src/tests/user_tests.rs
```

### Analyze impact

```
If I change the User struct, what would be affected?
```

Result:
```
Impact analysis for User:
- Directly affected: 8 files
- Transitively affected: 15 files
- Test files: 3
- Risk level: Medium
```

---

## Step 7: Create a Development Plan

Let's create a plan for some work you want to do:

```
Create a plan called "Add Password Reset Feature" with description
"Implement forgot password flow with email verification" and priority 10
```

Result:
```
Created plan:
- ID: abc-123-def
- Title: Add Password Reset Feature
- Status: draft
- Priority: 10
```

---

## Step 8: Add Tasks to the Plan

Break down the work into tasks:

```
Add these tasks to the plan:
1. "Design password reset API endpoints" with tags [design, api]
2. "Implement reset token generation" with tags [backend, security]
3. "Create password reset email template" with tags [email, frontend]
4. "Add reset password UI form" with tags [frontend, ui]
```

The AI will create each task:

```
Added 4 tasks to plan:
- Task 1: Design password reset API endpoints (pending)
- Task 2: Implement reset token generation (pending)
- Task 3: Create password reset email template (pending)
- Task 4: Add reset password UI form (pending)
```

### Set up dependencies

```
Task 2 should depend on Task 1.
Task 3 should depend on Task 2.
Task 4 should depend on Task 2.
```

Now tasks 3 and 4 won't be available until task 2 is done.

---

## Step 9: Work on a Task

Ask for your next task:

```
What's the next task I should work on?
```

Result:
```
Next task: Design password reset API endpoints
- Priority: 10
- Tags: design, api
- Status: pending
- Blocked by: nothing (ready to start)
```

Start working:

```
Mark that task as in progress and assign it to me
```

Result:
```
Task updated:
- Status: in_progress
- Assigned to: user
- Started at: 2024-01-15T10:30:00Z
```

---

## Step 10: Record Decisions

As you work, record important decisions:

```
Record a decision for this task:
Description: "Use JWT tokens for password reset instead of random strings"
Rationale: "JWTs can include expiration and user ID, making verification stateless"
Alternatives considered: "Random UUID tokens stored in database"
```

Result:
```
Decision recorded:
- ID: dec-456
- Task: Design password reset API endpoints
- Choice: JWT tokens for password reset
```

---

## Step 11: Complete the Task

When you're done:

```
Mark the task as completed
```

Result:
```
Task completed:
- Completed at: 2024-01-15T14:30:00Z
- Duration: 4 hours

Next available task: Implement reset token generation
(Previously blocked, now ready)
```

---

## Step 12: Track Progress

Check overall progress:

```
Show me the status of all tasks in this plan
```

Result:
```
Plan: Add Password Reset Feature
Progress: 1/4 tasks completed (25%)

Tasks:
✅ Design password reset API endpoints (completed)
⏳ Implement reset token generation (pending, unblocked)
🔒 Create password reset email template (pending, blocked by task 2)
🔒 Add reset password UI form (pending, blocked by task 2)
```

### View the dependency graph

```
Show me the dependency graph for this plan
```

Result:
```
Dependency Graph:
Task 1 (completed)
    └── Task 2 (pending)
            ├── Task 3 (blocked)
            └── Task 4 (blocked)
```

---

## What's Next?

You've learned the basics! Here's what to explore next:

### Advanced Features

- **[Multi-Agent Workflows](./multi-agent-workflow.md)** — Coordinate multiple AI agents
- **Milestones & Releases** — Group tasks for version planning
- **Constraints** — Add rules that must be followed (security, style, etc.)

### Deep Dives

- **[API Reference](../api/reference.md)** — Full REST API documentation
- **[MCP Tools Reference](../api/mcp-tools.md)** — All 137 tools explained

### Integration Guides

- **[Claude Code](../integrations/claude-code.md)** — Full Claude Code setup
- **[OpenAI](../integrations/openai.md)** — OpenAI Agents SDK integration
- **[Cursor](../integrations/cursor.md)** — Cursor IDE integration

---

## Quick Reference

### Common Commands

| Task | What to Ask |
|------|-------------|
| List projects | "List all projects" |
| Sync code | "Sync the {project} project" |
| Search code | "Search for {topic} in {project}" |
| Create plan | "Create a plan called {name}" |
| Add task | "Add task {description} to the plan" |
| Next task | "What's my next task?" |
| Update status | "Mark task as {status}" |
| Record decision | "Record a decision: {description}" |
| Check progress | "Show task progress for this plan" |

### Task Statuses

| Status | Meaning |
|--------|---------|
| `pending` | Not started, waiting |
| `in_progress` | Currently being worked on |
| `blocked` | Waiting on dependencies |
| `completed` | Done |
| `failed` | Couldn't complete |

---

## Troubleshooting

### "No projects found"

Make sure you've created and synced a project:
```
Create project my-app at /path/to/code
Sync the my-app project
```

### "Task is blocked"

Check what's blocking it:
```
What tasks are blocking {task_id}?
```

### "Search returns no results"

The project might not be synced. Run:
```
Sync the {project} project
```

### Backend connection errors

Check services are running:
```bash
docker compose ps
docker compose logs neo4j
docker compose logs meilisearch
```

---

Congratulations! You're now ready to use Project Orchestrator to coordinate your AI-assisted development workflow.
