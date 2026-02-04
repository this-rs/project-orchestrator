# Project Orchestrator

A Rust-based AI agent orchestrator with Neo4j knowledge graph, Meilisearch semantic search, and Tree-sitter code parsing.

## Overview

Project Orchestrator coordinates multiple AI coding agents working on complex projects by providing:

- **Shared Knowledge Base**: Code structure stored in Neo4j graph database
- **Semantic Search**: Fast code and decision search via Meilisearch
- **Plan Management**: Structured tasks with steps, constraints, and progress tracking
- **Auto-Sync**: File watcher keeps the knowledge base updated during development
- **Code Exploration API**: Query code relationships instead of reading files

## Quick Start

```bash
# 1. Start backends
docker compose up -d neo4j meilisearch

# 2. Build
cargo build --release

# 3. Run server
./target/release/orchestrator serve

# 4. Create a project and sync
curl -X POST http://localhost:8080/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project", "root_path": "/path/to/project"}'

curl -X POST http://localhost:8080/api/projects/my-project/sync
```

## Key Features

### Multi-Project Support

```bash
# Create a project
curl -X POST http://localhost:8080/api/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project", "root_path": "/path/to/project", "description": "Optional description"}'

# List projects
curl http://localhost:8080/api/projects

# Get project details
curl http://localhost:8080/api/projects/my-project

# Sync a project
curl -X POST http://localhost:8080/api/projects/my-project/sync

# Delete a project
curl -X DELETE http://localhost:8080/api/projects/my-project
```

### Plan & Task Management

Plans contain tasks with rich metadata, steps (subtasks), and constraints.

```bash
# Create a plan with constraints
curl -X POST http://localhost:8080/api/plans \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Implement Feature X",
    "description": "Add new authentication system",
    "priority": 10,
    "constraints": [
      {"constraint_type": "security", "description": "No plaintext passwords", "enforced_by": "code review"},
      {"constraint_type": "testing", "description": "100% test coverage", "enforced_by": "CI"}
    ]
  }'

# Add a rich task
curl -X POST http://localhost:8080/api/plans/{plan_id}/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Implement JWT authentication",
    "description": "Add JWT-based auth to all API endpoints",
    "priority": 9,
    "tags": ["backend", "security", "auth"],
    "acceptance_criteria": [
      "Login endpoint returns JWT token",
      "Protected routes require valid token",
      "Token expiration is enforced"
    ],
    "affected_files": ["src/api/auth.rs", "src/middleware/jwt.rs"],
    "estimated_complexity": 7
  }'

# Get plan details
curl http://localhost:8080/api/plans/{plan_id}

# Get next available task (highest priority, no blockers)
curl http://localhost:8080/api/plans/{plan_id}/next-task

# Update task status
curl -X PATCH http://localhost:8080/api/tasks/{task_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "in_progress", "assigned_to": "agent-1"}'
```

### Steps (Subtasks)

Break down tasks into ordered steps with verification criteria.

```bash
# Add steps to a task
curl -X POST http://localhost:8080/api/tasks/{task_id}/steps \
  -H "Content-Type: application/json" \
  -d '{"description": "Setup JWT library", "verification": "Can generate tokens"}'

curl -X POST http://localhost:8080/api/tasks/{task_id}/steps \
  -H "Content-Type: application/json" \
  -d '{"description": "Implement login endpoint", "verification": "POST /auth/login works"}'

# Get all steps for a task
curl http://localhost:8080/api/tasks/{task_id}/steps

# Update step status
curl -X PATCH http://localhost:8080/api/steps/{step_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'

# Get step progress
curl http://localhost:8080/api/tasks/{task_id}/steps/progress
# Returns: {"completed": 1, "total": 2, "percentage": 50.0}
```

### Constraints

Define constraints that must be respected throughout the plan.

```bash
# Add a constraint to a plan
curl -X POST http://localhost:8080/api/plans/{plan_id}/constraints \
  -H "Content-Type: application/json" \
  -d '{
    "constraint_type": "performance",
    "description": "Response time must be under 100ms",
    "enforced_by": "load tests"
  }'

# Get plan constraints
curl http://localhost:8080/api/plans/{plan_id}/constraints

# Delete a constraint
curl -X DELETE http://localhost:8080/api/constraints/{constraint_id}
```

Constraint types: `performance`, `security`, `compatibility`, `style`, `testing`, `other`

### Code Exploration

```bash
# Search code semantically
curl "http://localhost:8080/api/code/search?q=error+handling&limit=10"

# Get symbols in a file
curl "http://localhost:8080/api/code/symbols/src%2Flib.rs"

# Analyze change impact
curl "http://localhost:8080/api/code/impact?target=src/main.rs&target_type=file"

# View architecture overview
curl http://localhost:8080/api/code/architecture

# Find types implementing a trait
curl "http://localhost:8080/api/code/trait-impls?trait_name=Module"

# Find traits implemented by a type
curl "http://localhost:8080/api/code/type-traits?type_name=AppState"

# Get impl blocks for a type
curl "http://localhost:8080/api/code/impl-blocks?type_name=Orchestrator"

# Get function call graph
curl "http://localhost:8080/api/code/callgraph?function=main&depth=2&direction=both"

# Find similar code
curl -X POST http://localhost:8080/api/code/similar \
  -H "Content-Type: application/json" \
  -d '{"snippet": "async fn handle_request", "limit": 5}'
```

### Decision Tracking

Record and search architectural decisions made during development.

```bash
# Record a decision
curl -X POST http://localhost:8080/api/tasks/{task_id}/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Use JWT instead of session cookies",
    "rationale": "Better for stateless API, easier horizontal scaling",
    "alternatives": ["Session cookies", "OAuth tokens"],
    "chosen_option": "JWT with refresh tokens"
  }'

# Search past decisions
curl "http://localhost:8080/api/decisions/search?q=authentication&limit=10"
```

### File Watcher (Auto-Sync)

```bash
# Start watching a directory
curl -X POST http://localhost:8080/api/watch \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/project"}'

# Check watcher status
curl http://localhost:8080/api/watch

# Stop watching
curl -X DELETE http://localhost:8080/api/watch
```

### Agent Webhooks

Agents can report task completion via webhooks.

```bash
# Report task completion
curl -X POST http://localhost:8080/api/wake \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "uuid-here",
    "success": true,
    "summary": "Implemented JWT authentication",
    "files_modified": ["src/api/auth.rs", "src/middleware/jwt.rs"]
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR API                          │
│                    (localhost:8080)                          │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Projects │  │  Plans   │  │  Code    │  │  Watch   │    │
│  │  API     │  │Tasks/Steps│ │  API     │  │  API     │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    NEO4J      │     │  MEILISEARCH  │     │  TREE-SITTER  │
│   (7687)      │     │    (7700)     │     │   (in-proc)   │
│               │     │               │     │               │
│ • Code graph  │     │ • Code search │     │ • AST parsing │
│ • Projects    │     │ • Decisions   │     │ • Symbols     │
│ • Plans/Tasks │     │               │     │ • Complexity  │
│ • Steps       │     │               │     │ • Impl blocks │
│ • Constraints │     │               │     │ • Call graph  │
│ • Decisions   │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
```

## Data Model

### Task Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `title` | string? | Short title |
| `description` | string | Detailed description |
| `status` | enum | `pending`, `in_progress`, `blocked`, `completed`, `failed` |
| `priority` | int? | Higher = more important |
| `tags` | string[] | Labels for categorization |
| `acceptance_criteria` | string[] | Conditions for completion |
| `affected_files` | string[] | Files expected to be modified |
| `estimated_complexity` | int? | 1-10 scale |
| `assigned_to` | string? | Agent ID |
| `depends_on` | UUID[] | Task dependencies |

### Step Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `order` | int | Execution order (0-based) |
| `description` | string | What to do |
| `status` | enum | `pending`, `in_progress`, `completed`, `skipped` |
| `verification` | string? | How to verify completion |

### Constraint Types

| Type | Description |
|------|-------------|
| `performance` | Response time, throughput requirements |
| `security` | Security requirements and restrictions |
| `compatibility` | API/ABI compatibility constraints |
| `style` | Code style and conventions |
| `testing` | Test coverage requirements |
| `other` | Other constraints |

## Tests

```bash
# Run all tests (requires server running on port 8080)
./target/release/orchestrator serve &
cargo test

# Expected: 62 tests passing
# - 2 unit tests (watcher, slugify)
# - 29 API tests
# - 8 integration tests
# - 23 parser tests

# Run specific test suites
cargo test --test parser_tests      # Code parsing
cargo test --test integration_tests # Database operations
cargo test --test api_tests         # HTTP API
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `orchestrator123` | Neo4j password |
| `MEILISEARCH_URL` | `http://localhost:7700` | Meilisearch URL |
| `MEILISEARCH_KEY` | `orchestrator-meili-key-change-me` | Meilisearch key |
| `SERVER_PORT` | `8080` | API server port |

## Supported Languages

The Tree-sitter parser extracts code structure from:

- **Rust** (`.rs`) - Full support: structs, traits, enums, impl blocks, functions, imports
- **TypeScript/JavaScript** (`.ts`, `.tsx`, `.js`, `.jsx`) - Functions, classes, imports
- **Python** (`.py`) - Functions, classes, imports
- **Go** (`.go`) - Functions, structs, interfaces

## API Reference

### Projects
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects` | List all projects |
| POST | `/api/projects` | Create project |
| GET | `/api/projects/{slug}` | Get project details |
| DELETE | `/api/projects/{slug}` | Delete project |
| POST | `/api/projects/{slug}/sync` | Sync project files |
| GET | `/api/projects/{slug}/plans` | List project plans |
| GET | `/api/projects/{slug}/code/search` | Search project code |

### Plans & Tasks
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/plans` | List active plans |
| POST | `/api/plans` | Create plan |
| GET | `/api/plans/{id}` | Get plan details |
| PATCH | `/api/plans/{id}` | Update plan status |
| GET | `/api/plans/{id}/next-task` | Get next available task |
| POST | `/api/plans/{id}/tasks` | Add task to plan |
| GET | `/api/tasks/{id}` | Get task details |
| PATCH | `/api/tasks/{id}` | Update task |

### Steps
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tasks/{id}/steps` | List task steps |
| POST | `/api/tasks/{id}/steps` | Add step |
| GET | `/api/tasks/{id}/steps/progress` | Get completion progress |
| PATCH | `/api/steps/{id}` | Update step |

### Constraints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/plans/{id}/constraints` | List plan constraints |
| POST | `/api/plans/{id}/constraints` | Add constraint |
| DELETE | `/api/constraints/{id}` | Delete constraint |

### Code Exploration
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/code/search` | Semantic code search |
| GET | `/api/code/symbols/{path}` | Get file symbols |
| GET | `/api/code/references` | Find symbol references |
| GET | `/api/code/dependencies/{path}` | Get file dependencies |
| GET | `/api/code/callgraph` | Get function call graph |
| GET | `/api/code/impact` | Analyze change impact |
| GET | `/api/code/architecture` | Get architecture overview |
| POST | `/api/code/similar` | Find similar code |
| GET | `/api/code/trait-impls` | Find trait implementations |
| GET | `/api/code/type-traits` | Find type's traits |
| GET | `/api/code/impl-blocks` | Get impl blocks |

### Other
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/sync` | Sync directory |
| GET/POST/DELETE | `/api/watch` | File watcher |
| POST | `/api/tasks/{id}/decisions` | Record decision |
| GET | `/api/decisions/search` | Search decisions |
| POST | `/api/wake` | Agent webhook |

## License

MIT
