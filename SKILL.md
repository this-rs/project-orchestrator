---
name: project-orchestrator
description: AI agent orchestrator with Neo4j knowledge graph, Meilisearch search, and Tree-sitter parsing. Use for coordinating multiple coding agents on complex projects with shared context and plans.
metadata:
  clawdbot:
    emoji: "🎯"
    requires:
      bins: ["docker", "cargo"]
---

# Project Orchestrator

Coordinate multiple AI coding agents with a shared knowledge base.

## Features

- **Multi-Project Support**: Manage multiple codebases with isolated data
- **Neo4j Knowledge Graph**: Code structure, relationships, plans, decisions
- **Meilisearch**: Fast semantic search across code and decisions
- **Tree-sitter**: Precise code parsing for 12 languages
- **Plan Management**: Structured tasks with dependencies and constraints
- **MCP Integration**: 62 tools for Claude Code, OpenAI Agents, and Cursor

## Documentation

- [Installation Guide](docs/setup/installation.md)
- [Getting Started Tutorial](docs/guides/getting-started.md)
- [API Reference](docs/api/reference.md)
- [MCP Tools Reference](docs/api/mcp-tools.md)
- Integration Guides: [Claude Code](docs/integrations/claude-code.md) | [OpenAI](docs/integrations/openai.md) | [Cursor](docs/integrations/cursor.md)

## Quick Start

### 1. Start the backends

```bash
cd {baseDir}
docker compose up -d neo4j meilisearch
```

### 2. Build and run the orchestrator

```bash
cargo build --release
./target/release/orchestrator serve
```

Or with Docker:
```bash
docker compose up -d
```

### 3. Sync your codebase

```bash
# Via CLI
./target/release/orch sync --path /path/to/project

# Via API
curl -X POST http://localhost:8080/api/sync \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project"}'
```

## Usage

### Create a project

```bash
# Create a new project
curl -X POST http://localhost:8080/api/projects \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Embryon",
    "root_path": "/Users/triviere/projects/embryon",
    "description": "Neural network composition framework"
  }'

# List all projects
curl http://localhost:8080/api/projects

# Sync a project
curl -X POST http://localhost:8080/api/projects/embryon/sync

# Search code within a project
curl "http://localhost:8080/api/projects/embryon/code/search?q=tensor&limit=10"
```

### Create a plan

```bash
orch plan create \
  --title "Implement GPU Backend" \
  --desc "Add Metal GPU support for neural network operations" \
  --priority 10
```

### Add tasks to the plan

```bash
orch task add \
  --plan <plan-id> \
  --desc "Implement MatMul Metal shader"

orch task add \
  --plan <plan-id> \
  --desc "Add attention layer GPU support" \
  --depends <task-1-id>
```

### Get context for an agent

```bash
# JSON context
orch context --plan <plan-id> --task <task-id>

# Ready-to-use prompt
orch context --plan <plan-id> --task <task-id> --prompt
```

### Record decisions

```bash
orch decision add \
  --task <task-id> \
  --desc "Use shared memory for tile-based MatMul" \
  --rationale "Better cache locality, 2x performance improvement"
```

### Search past decisions

```bash
orch decision search "memory management GPU"
```

## API Endpoints

### Projects (Multi-Project Support)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create a new project |
| GET | `/api/projects/{slug}` | Get project by slug |
| DELETE | `/api/projects/{slug}` | Delete a project |
| POST | `/api/projects/{slug}/sync` | Sync project's codebase |
| GET | `/api/projects/{slug}/plans` | List project's plans |
| GET | `/api/projects/{slug}/code/search` | Search code in project |

### Plans & Tasks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/plans` | List active plans |
| POST | `/api/plans` | Create plan |
| GET | `/api/plans/{id}` | Get plan details |
| PATCH | `/api/plans/{id}` | Update plan status |
| GET | `/api/plans/{id}/next-task` | Get next available task |
| POST | `/api/plans/{id}/tasks` | Add task to plan |
| GET | `/api/tasks/{id}` | Get task details |
| PATCH | `/api/tasks/{id}` | Update task |
| GET | `/api/plans/{plan}/tasks/{task}/context` | Get task context |
| GET | `/api/plans/{plan}/tasks/{task}/prompt` | Get generated prompt |
| POST | `/api/tasks/{id}/decisions` | Add decision |
| GET | `/api/decisions/search?q=...` | Search decisions |

### Sync & Watch

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/sync` | Sync directory to knowledge base |
| GET | `/api/watch` | Get file watcher status |
| POST | `/api/watch` | Start watching a directory |
| DELETE | `/api/watch` | Stop file watcher |
| POST | `/api/wake` | Agent completion webhook |

### Code Exploration (Graph + Search)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/code/search?q=...` | Semantic code search |
| GET | `/api/code/symbols/{path}` | Get symbols in a file |
| GET | `/api/code/references?symbol=...` | Find all references to a symbol |
| GET | `/api/code/dependencies/{path}` | Get file import/dependent graph |
| GET | `/api/code/callgraph?function=...` | Get function call graph |
| GET | `/api/code/impact?target=...` | Analyze change impact |
| GET | `/api/code/architecture` | Get codebase overview |
| POST | `/api/code/similar` | Find similar code snippets |
| GET | `/api/code/trait-impls?trait_name=...` | Find types implementing a trait |
| GET | `/api/code/type-traits?type_name=...` | Find traits implemented by a type |
| GET | `/api/code/impl-blocks?type_name=...` | Get all impl blocks for a type |

## Auto-Sync with File Watcher

Keep the knowledge base updated automatically while coding:

```bash
# Start watching a project directory
curl -X POST http://localhost:8080/api/watch \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project"}'

# Check watcher status
curl http://localhost:8080/api/watch

# Stop watching
curl -X DELETE http://localhost:8080/api/watch
```

The watcher automatically syncs `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.go` files when modified.
It ignores `node_modules/`, `target/`, `.git/`, `__pycache__/`, `dist/`, `build/`.

## Code Exploration

Query the code graph instead of reading files directly:

```bash
# Semantic search across code
curl "http://localhost:8080/api/code/search?q=error+handling&language=rust&limit=10"

# Get symbols in a file (functions, structs, etc.)
curl "http://localhost:8080/api/code/symbols/src%2Flib.rs"

# Find all references to a symbol
curl "http://localhost:8080/api/code/references?symbol=AppState&limit=20"

# Get file dependencies (imports and dependents)
curl "http://localhost:8080/api/code/dependencies/src%2Fneo4j%2Fclient.rs"

# Get call graph for a function
curl "http://localhost:8080/api/code/callgraph?function=handle_request&depth=2&direction=both"

# Analyze impact before changing a file
curl "http://localhost:8080/api/code/impact?target=src/lib.rs&target_type=file"

# Get architecture overview
curl "http://localhost:8080/api/code/architecture"

# Find similar code patterns
curl -X POST http://localhost:8080/api/code/similar \
  -H "Content-Type: application/json" \
  -d '{"snippet": "async fn handle_error", "limit": 5}'

# Find all types implementing a trait
curl "http://localhost:8080/api/code/trait-impls?trait_name=Module"

# Find all traits implemented by a type
curl "http://localhost:8080/api/code/type-traits?type_name=Orchestrator"

# Get all impl blocks for a type
curl "http://localhost:8080/api/code/impl-blocks?type_name=Neo4jClient"
```

## For Agents

### Getting context before starting work

```bash
# Fetch your task context
curl http://localhost:8080/api/plans/$PLAN_ID/tasks/$TASK_ID/prompt
```

### Recording decisions while working

```bash
curl -X POST http://localhost:8080/api/tasks/$TASK_ID/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Chose X over Y",
    "rationale": "Because..."
  }'
```

### Notifying completion

```bash
curl -X POST http://localhost:8080/api/wake \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "'$TASK_ID'",
    "success": true,
    "summary": "Implemented feature X",
    "files_modified": ["src/foo.rs", "src/bar.rs"]
  }'
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `orchestrator123` | Neo4j password |
| `MEILISEARCH_URL` | `http://localhost:7700` | Meilisearch URL |
| `MEILISEARCH_KEY` | `orchestrator-meili-key-change-me` | Meilisearch API key |
| `WORKSPACE_PATH` | `.` | Default workspace path |
| `SERVER_PORT` | `8080` | Server port |
| `RUST_LOG` | `info` | Log level |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR API                          │
│                    (localhost:8080)                          │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    NEO4J      │     │  MEILISEARCH  │     │  TREE-SITTER  │
│   (7687)      │     │    (7700)     │     │   (in-proc)   │
│               │     │               │     │               │
│ • Code graph  │     │ • Code search │     │ • AST parsing │
│ • Plans       │     │ • Decisions   │     │ • Symbols     │
│ • Decisions   │     │ • Logs        │     │ • Complexity  │
│ • Relations   │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Development

```bash
# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run -- serve

# Format code
cargo fmt

# Lint
cargo clippy
```
