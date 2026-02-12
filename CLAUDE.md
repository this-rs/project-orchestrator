# CLAUDE.md

This file provides guidance to Claude Code when working on the project-orchestrator skill.

## Documentation

For user-facing documentation, see the `docs/` folder:

- **[Installation Guide](docs/setup/installation.md)** — Full setup instructions
- **[Getting Started](docs/guides/getting-started.md)** — Tutorial for new users
- **[API Reference](docs/api/reference.md)** — REST API documentation
- **[MCP Tools](docs/api/mcp-tools.md)** — All 145 MCP tools documented
- **Integration Guides:**
  - [Claude Code](docs/integrations/claude-code.md)
  - [OpenAI Agents](docs/integrations/openai.md)
  - [Cursor](docs/integrations/cursor.md)
- **[Multi-Agent Workflows](docs/guides/multi-agent-workflow.md)** — Advanced coordination
- **[Knowledge Notes](docs/guides/knowledge-notes.md)** — Contextual knowledge capture system
- **[Workspaces](docs/guides/workspaces.md)** — Multi-project workspace coordination
- **[Authentication](docs/guides/authentication.md)** — JWT + Google OAuth setup
- **[Chat & WebSocket](docs/guides/chat-websocket.md)** — Real-time chat and events

## Project Overview

**Project Orchestrator** is a Rust-based service that coordinates AI coding agents on complex projects. It provides:

- Neo4j graph database for code structure and relationships
- Meilisearch for semantic search across code and decisions
- Tree-sitter for multi-language code parsing
- HTTP API for plans, tasks, decisions, and code exploration
- MCP server for Claude Code integration (145 tools)
- File watcher for auto-syncing changes
- Authentication system: Google OAuth2, generic OIDC, password login + JWT, deny-by-default middleware
- Chat WebSocket for real-time conversational AI (migrated from SSE)
- Event system: live CRUD notifications via WebSocket
- YAML configuration system with env var overrides (priority: env > yaml > default)

## Build Commands

```bash
cargo build --release          # Build release binary
cargo test                     # Run all tests (752 total, mock backends)
cargo clippy                   # Lint
cargo fmt                      # Format
```

## Running the Server

```bash
# Start backends first
docker compose up -d neo4j meilisearch

# Run server
./target/release/orchestrator serve --port 8080

# Or with debug logging
RUST_LOG=debug ./target/release/orchestrator serve
```

## MCP Server (Claude Code Integration)

The project-orchestrator can run as an MCP (Model Context Protocol) server, exposing all orchestrator functionality as tools for Claude Code.

### Building the MCP Server

```bash
cargo build --release --bin mcp_server
```

The binary will be at `./target/release/mcp_server`.

### Configuring Claude Code

Add to your Claude Code MCP settings (`~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "MEILISEARCH_URL": "http://localhost:7700",
        "MEILISEARCH_KEY": "your-meilisearch-key"
      }
    }
  }
}
```

Or use command-line arguments:

```json
{
  "mcpServers": {
    "project-orchestrator": {
      "command": "/path/to/mcp_server",
      "args": [
        "--neo4j-uri", "bolt://localhost:7687",
        "--neo4j-user", "neo4j",
        "--neo4j-password", "your-password",
        "--meilisearch-url", "http://localhost:7700",
        "--meilisearch-key", "your-key"
      ]
    }
  }
}
```

### Available MCP Tools

The MCP server exposes 145 tools organized by category:

**Project Management (8 tools)**
- `list_projects` - List all registered projects
- `get_project` - Get project details by slug
- `register_project` - Register a new project
- `sync_project` - Sync project files to knowledge graph
- `search_code` - Semantic search across code
- `get_project_stats` - Get project statistics
- `get_roadmap` - Get project roadmap with milestones/releases
- `unregister_project` - Remove a project

**Plan Management (12 tools)**
- `list_plans` - List all plans with filtering
- `create_plan` - Create a new plan
- `get_plan` - Get plan details
- `update_plan` - Update plan status/details
- `link_plan_to_project` - Associate plan with project
- `get_plan_context` - Get rich context for agent work
- `get_dependency_graph` - Get task dependency visualization
- `get_critical_path` - Find longest dependency chain
- `add_constraint` - Add plan constraint
- `list_constraints` - List plan constraints
- `remove_constraint` - Remove a constraint
- `generate_prompt` - Generate agent prompt with context

**Task Management (16 tools)**
- `list_tasks` - List tasks with filtering
- `create_task` - Create a new task
- `get_task` - Get task details
- `update_task` - Update task
- `get_next_task` - Get next available task
- `add_task_dependencies` - Add dependencies to task
- `remove_task_dependency` - Remove a dependency
- `get_task_blockers` - Get blocking tasks
- `get_blocked_tasks` - Get tasks blocked by this one
- `create_step` - Add step to task
- `list_steps` - List task steps
- `update_step` - Update step status
- `get_step_progress` - Get step completion progress
- `record_decision` - Record a decision
- `list_decisions` - List decisions for a task
- `assign_task` - Assign task to agent

**Commit Tracking (5 tools)**
- `create_commit` - Register a commit
- `link_commit_to_task` - Link commit to task
- `link_commit_to_plan` - Link commit to plan
- `get_task_commits` - Get commits for task
- `get_plan_commits` - Get commits for plan

**Release Management (6 tools)**
- `create_release` - Create a release
- `list_releases` - List project releases
- `get_release` - Get release details
- `update_release` - Update release
- `add_task_to_release` - Add task to release
- `add_commit_to_release` - Add commit to release

**Milestone Management (6 tools)**
- `create_milestone` - Create a milestone
- `list_milestones` - List project milestones
- `get_milestone` - Get milestone details
- `update_milestone` - Update milestone
- `add_task_to_milestone` - Add task to milestone
- `get_milestone_progress` - Get completion progress

**Code Exploration (9 tools)**
- `get_file_symbols` - Get symbols in a file
- `find_symbol_references` - Find references to a symbol
- `get_file_dependencies` - Get file imports/dependents
- `get_call_graph` - Get function call graph
- `analyze_impact` - Analyze change impact
- `get_codebase_overview` - Get architecture overview
- `find_similar_code` - Find similar code snippets
- `find_trait_implementations` - Find trait implementations
- `get_type_traits` - Get traits for a type

**Workspace Management (9 tools)**
- `list_workspaces` - List all workspaces
- `create_workspace` - Create workspace to group projects
- `get_workspace` - Get workspace details
- `update_workspace` - Update workspace
- `delete_workspace` - Delete workspace
- `get_workspace_overview` - Overview with projects, milestones, resources
- `list_workspace_projects` - List projects in workspace
- `add_project_to_workspace` - Add project to workspace
- `remove_project_from_workspace` - Remove project from workspace

**Workspace Milestones (7 tools)**
- `list_workspace_milestones` - List cross-project milestones
- `create_workspace_milestone` - Create cross-project milestone
- `get_workspace_milestone` - Get milestone details
- `update_workspace_milestone` - Update milestone
- `delete_workspace_milestone` - Delete milestone
- `add_task_to_workspace_milestone` - Add task from any project
- `get_workspace_milestone_progress` - Get completion progress

**Resources (5 tools)**
- `list_resources` - List shared resources (API contracts, schemas)
- `create_resource` - Create resource reference
- `get_resource` - Get resource details
- `delete_resource` - Delete resource
- `link_resource_to_project` - Link resource (implements/uses)

**Components & Topology (8 tools)**
- `list_components` - List workspace components
- `create_component` - Create component (Service, Database, etc.)
- `get_component` - Get component details
- `delete_component` - Delete component
- `add_component_dependency` - Add component dependency
- `remove_component_dependency` - Remove dependency
- `map_component_to_project` - Map component to project
- `get_workspace_topology` - Get full topology graph

### Debug Logging

Enable debug logging with `RUST_LOG=debug`:

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

## Project Structure

```
docs/
├── setup/
│   └── installation.md      # Installation and configuration
├── integrations/
│   ├── claude-code.md       # Claude Code MCP setup
│   ├── openai.md            # OpenAI Agents SDK setup
│   └── cursor.md            # Cursor IDE setup
├── api/
│   ├── reference.md         # REST API documentation
│   └── mcp-tools.md         # MCP tools reference (145 tools)
└── guides/
    ├── getting-started.md   # Tutorial for new users
    ├── multi-agent-workflow.md # Multi-agent coordination
    ├── knowledge-notes.md   # Knowledge Notes system guide
    ├── authentication.md    # JWT + Google OAuth guide
    └── chat-websocket.md    # Chat and events guide

src/
├── auth/
│   └── ...              # Authentication: JWT (HS256), Google OAuth2, middleware, extractors
├── config/
│   └── ...              # YAML configuration system with env var overrides
├── events/
│   └── ...              # EventBus broadcast, EventEmitter trait, NATS emitter, HybridEmitter, WebSocket notifications
├── api/
│   ├── mod.rs           # API module exports
│   ├── routes.rs        # Route definitions (axum)
│   ├── handlers.rs      # Plan/Task/Decision handlers
│   ├── auth_handlers.rs # Google OAuth + JWT endpoints
│   ├── chat_handlers.rs # Chat WebSocket endpoints
│   ├── ws_auth.rs       # WebSocket authentication helpers
│   ├── ws_chat_handler.rs # Chat WebSocket handler (local/NATS/resume routing)
│   ├── project_handlers.rs # Project-specific endpoints
│   ├── embedded_frontend.rs # Static SPA serving (rust-embed)
│   ├── code_handlers.rs # Code exploration endpoints
│   ├── note_handlers.rs # Knowledge Notes endpoints
│   └── workspace_handlers.rs # Workspace endpoints
├── chat/
│   ├── mod.rs           # Chat module exports
│   ├── config.rs        # ChatConfig (MCP server path, model, timeouts)
│   ├── manager.rs       # ChatManager — Nexus SDK orchestration
│   └── types.rs         # ChatRequest, ChatEvent, ChatSession, ClientMessage
├── mcp/
│   ├── mod.rs           # MCP module exports
│   ├── protocol.rs      # JSON-RPC 2.0 types
│   ├── tools.rs         # Tool definitions (145 tools)
│   ├── handlers.rs      # Tool implementations
│   └── server.rs        # MCP server (stdio)
├── neo4j/
│   ├── client.rs        # Neo4j connection and queries
│   ├── models.rs        # Graph node types
│   ├── traits.rs        # GraphStore trait (178 methods)
│   ├── impl_graph_store.rs # Neo4j implementation
│   └── mock.rs          # MockGraphStore for testing (#[cfg(test)])
├── meilisearch/
│   ├── client.rs        # Meilisearch connection
│   ├── models.rs        # Search document types
│   ├── traits.rs        # SearchStore trait (24 methods)
│   ├── impl_search_store.rs # Meilisearch implementation
│   └── mock.rs          # MockSearchStore for testing (#[cfg(test)])
├── notes/
│   ├── mod.rs           # Notes module exports
│   ├── models.rs        # Note types (NoteType, NoteStatus, etc.)
│   ├── manager.rs       # NoteManager CRUD operations
│   ├── lifecycle.rs     # Staleness calculation, obsolescence detection
│   └── hashing.rs       # Semantic hashing for code anchors
├── parser/
│   ├── mod.rs           # CodeParser, SupportedLanguage, dispatch
│   ├── helpers.rs       # Shared utility functions
│   └── languages/       # Per-language extractors (12 languages)
├── plan/
│   ├── manager.rs       # Plan/Task CRUD operations
│   └── models.rs        # Plan/Task/Decision types
├── orchestrator/
│   ├── runner.rs        # Main orchestrator logic
│   ├── context.rs       # Agent context builder (includes notes)
│   └── watcher.rs       # File watcher for auto-sync
├── bin/
│   └── mcp_server.rs    # MCP server binary
├── test_helpers.rs      # mock_app_state(), test_project(), test_plan()...
├── lib.rs               # Library exports
├── setup_claude.rs      # Auto-configure Claude Code MCP (setup-claude command)
├── update.rs            # Self-update via GitHub Releases (update command)
└── main.rs              # CLI entry point

tests/
├── api_tests.rs         # HTTP API tests (29)
├── integration_tests.rs # Database tests (8)
└── parser_tests.rs      # Parser tests (33)
```

## Key APIs

### Plans & Tasks
- `GET /api/plans` - List plans with pagination and filters (see Query Parameters below)
- `POST /api/plans` - Create plan (with optional `project_id` to associate with project)
- `GET /api/plans/{id}` - Get plan details
- `PUT /api/plans/{id}/project` - Link plan to a project
- `DELETE /api/plans/{id}/project` - Unlink plan from project
- `GET /api/plans/{id}/next-task` - Get next available task
- `GET /api/plans/{id}/dependency-graph` - Get task dependency graph for visualization
- `GET /api/plans/{id}/critical-path` - Get longest dependency chain
- `POST /api/plans/{id}/tasks` - Add task (with title, priority, tags, acceptance_criteria, affected_files)
- `GET /api/tasks` - List all tasks across plans with pagination and filters (see Query Parameters below)
- `GET /api/tasks/{id}` - Get task details
- `PATCH /api/tasks/{id}` - Update task
- `POST /api/tasks/{id}/decisions` - Record decision

### Task Dependencies
- `POST /api/tasks/{id}/dependencies` - Add dependencies after task creation
- `DELETE /api/tasks/{id}/dependencies/{dep_id}` - Remove a dependency
- `GET /api/tasks/{id}/blockers` - Get tasks blocking this task (uncompleted dependencies)
- `GET /api/tasks/{id}/blocking` - Get tasks blocked by this task

### Commits
- `POST /api/commits` - Create/register a commit
- `GET /api/tasks/{id}/commits` - Get commits linked to a task
- `POST /api/tasks/{id}/commits` - Link a commit to a task (RESOLVED_BY)
- `GET /api/plans/{id}/commits` - Get commits linked to a plan
- `POST /api/plans/{id}/commits` - Link a commit to a plan (RESULTED_IN)

### Releases
- `POST /api/projects/{id}/releases` - Create a release for a project
- `GET /api/projects/{id}/releases` - List project releases
- `GET /api/releases/{id}` - Get release details with tasks and commits
- `PATCH /api/releases/{id}` - Update release (status, dates, title, description)
- `POST /api/releases/{id}/tasks` - Add task to release
- `POST /api/releases/{id}/commits` - Add commit to release

### Milestones
- `POST /api/projects/{id}/milestones` - Create a milestone for a project
- `GET /api/projects/{id}/milestones` - List project milestones
- `GET /api/milestones/{id}` - Get milestone details with tasks
- `PATCH /api/milestones/{id}` - Update milestone (status, dates, title, description)
- `POST /api/milestones/{id}/tasks` - Add task to milestone
- `GET /api/milestones/{id}/progress` - Get completion percentage

### Roadmap
- `GET /api/projects/{id}/roadmap` - Get aggregated roadmap view with:
  - Milestones ordered by target_date with tasks and progress
  - Releases with status, tasks, and commits
  - Project-wide progress statistics
  - Full dependency graph across all tasks

### Steps (Subtasks)
- `GET /api/tasks/{id}/steps` - Get task steps
- `POST /api/tasks/{id}/steps` - Add step to task
- `PATCH /api/steps/{id}` - Update step status
- `GET /api/tasks/{id}/steps/progress` - Get step completion progress

### Constraints
- `GET /api/plans/{id}/constraints` - Get plan constraints
- `POST /api/plans/{id}/constraints` - Add constraint (performance, security, style, etc.)
- `DELETE /api/constraints/{id}` - Remove constraint

### Code Exploration
- `GET /api/code/search?q=...` - Semantic search (with ranking scores)
- `GET /api/code/symbols/{path}` - File symbols (functions, structs, imports with details)
- `GET /api/code/references?symbol=...` - Find all references to a symbol
- `GET /api/code/dependencies/{path}` - File imports and dependents
- `GET /api/code/callgraph?function=...` - Function call graph
- `GET /api/code/impact?target=...` - Change impact analysis (uses IMPORTS relationships)
- `GET /api/code/architecture` - Codebase overview (most connected files)
- `GET /api/code/similar` - Find similar code (POST with snippet)
- `GET /api/code/trait-impls?trait_name=...` - Find trait implementations
- `GET /api/code/type-traits?type_name=...` - Find traits for a type
- `GET /api/code/impl-blocks?type_name=...` - Get impl blocks for a type

### Knowledge Notes
See the [Knowledge Notes Guide](docs/guides/knowledge-notes.md) for detailed documentation.

- `GET /api/notes` - List notes with filters (project_id, note_type, status, importance, tags)
- `POST /api/notes` - Create note (project_id, note_type, content, importance, tags)
- `GET /api/notes/{id}` - Get note details
- `PATCH /api/notes/{id}` - Update note (content, importance, status, tags)
- `DELETE /api/notes/{id}` - Delete note
- `GET /api/notes/search?q=...` - Semantic search across notes
- `GET /api/notes/context` - Get notes for entity (direct + propagated via graph)
- `GET /api/notes/needs-review` - List stale/needs_review notes
- `POST /api/notes/update-staleness` - Recalculate staleness scores
- `POST /api/notes/{id}/confirm` - Confirm note validity (reset staleness)
- `POST /api/notes/{id}/invalidate` - Mark note as obsolete
- `POST /api/notes/{id}/supersede` - Replace with new note
- `POST /api/notes/{id}/links` - Link note to entity
- `DELETE /api/notes/{id}/links/{type}/{entity}` - Unlink note from entity
- `GET /api/projects/{id}/notes` - List notes for a project

**Note Types:** `guideline`, `gotcha`, `pattern`, `context`, `tip`, `observation`, `assertion`

**Note Status:** `active`, `needs_review`, `stale`, `obsolete`, `archived`

**Importance Levels:** `critical`, `high`, `medium`, `low`

### Workspaces

Workspaces group related projects and provide shared context (cross-project milestones, resources, components).

**Workspace CRUD:**
- `GET /api/workspaces` - List workspaces with search/pagination
- `POST /api/workspaces` - Create workspace
- `GET /api/workspaces/{slug}` - Get workspace by slug
- `PATCH /api/workspaces/{slug}` - Update workspace
- `DELETE /api/workspaces/{slug}` - Delete workspace
- `GET /api/workspaces/{slug}/overview` - Overview with projects, milestones, resources, progress

**Workspace-Project Association:**
- `GET /api/workspaces/{slug}/projects` - List projects in workspace
- `POST /api/workspaces/{slug}/projects` - Add project to workspace
- `DELETE /api/workspaces/{slug}/projects/{id}` - Remove project from workspace

**Workspace Milestones (cross-project):**
- `GET /api/workspaces/{slug}/milestones` - List workspace milestones
- `POST /api/workspaces/{slug}/milestones` - Create workspace milestone
- `GET /api/workspace-milestones/{id}` - Get milestone with tasks
- `PATCH /api/workspace-milestones/{id}` - Update milestone
- `DELETE /api/workspace-milestones/{id}` - Delete milestone
- `POST /api/workspace-milestones/{id}/tasks` - Add task from any project
- `GET /api/workspace-milestones/{id}/progress` - Get completion progress

**Resources (shared contracts/specs):**
- `GET /api/workspaces/{slug}/resources` - List resources
- `POST /api/workspaces/{slug}/resources` - Create resource reference
- `GET /api/resources/{id}` - Get resource details
- `DELETE /api/resources/{id}` - Delete resource
- `POST /api/resources/{id}/projects` - Link project (implements/uses)

**Components & Topology:**
- `GET /api/workspaces/{slug}/components` - List components
- `POST /api/workspaces/{slug}/components` - Create component
- `GET /api/components/{id}` - Get component
- `DELETE /api/components/{id}` - Delete component
- `POST /api/components/{id}/dependencies` - Add dependency
- `DELETE /api/components/{id}/dependencies/{dep_id}` - Remove dependency
- `PUT /api/components/{id}/project` - Map to project
- `GET /api/workspaces/{slug}/topology` - Full topology graph

**Resource Types:** `ApiContract`, `Protobuf`, `GraphqlSchema`, `JsonSchema`, `DatabaseSchema`, `SharedTypes`, `Config`, `Documentation`, `Other`

**Component Types:** `Service`, `Frontend`, `Worker`, `Database`, `MessageQueue`, `Cache`, `Gateway`, `External`, `Other`

### Query Parameters (Pagination & Filtering)

List endpoints (`GET /api/plans`, `GET /api/tasks`, `GET /api/projects/{id}/releases`, `GET /api/projects/{id}/milestones`, `GET /api/projects`, `GET /api/workspaces`) support:

**Pagination:**
- `limit` - Max items per page (default: 50, max: 100)
- `offset` - Items to skip (default: 0)
- `sort_by` - Field to sort by (e.g., "created_at", "priority", "title")
- `sort_order` - Sort direction: "asc" or "desc" (default: "desc")

**Filtering:**
- `status` - Comma-separated status values (e.g., "pending,in_progress")
- `priority_min` - Minimum priority (inclusive)
- `priority_max` - Maximum priority (inclusive)
- `search` - Search in title/description (plans, projects)
- `tags` - Comma-separated tags (tasks only)
- `assigned_to` - Filter by assigned agent (tasks only)
- `plan_id` - Filter by plan ID (tasks only)

**Response format (PaginatedResponse):**
```json
{
  "items": [...],
  "total": 42,
  "limit": 50,
  "offset": 0,
  "has_more": false
}
```

**Examples:**
```bash
# Plans paginés avec filtres
GET /api/plans?status=draft,in_progress&priority_min=5&limit=10&offset=0

# Toutes les tâches d'un agent
GET /api/tasks?assigned_to=agent-1&status=in_progress

# Milestones actifs
GET /api/projects/{id}/milestones?status=open&limit=5

# Recherche projets
GET /api/projects?search=orchestrator&limit=10
```

### Chat (WebSocket)

Conversational interface with Claude Code CLI via Nexus SDK. Uses WebSocket for real-time streaming.

**Session lifecycle:**
- `POST /api/chat/sessions` — Create session + send first message → `{ session_id, stream_url }`
- `GET /ws/chat/{session_id}` — WebSocket connection for real-time streaming
- `POST /api/chat/sessions/{id}/messages` — Send follow-up message (auto-resumes inactive sessions)
- `POST /api/chat/sessions/{id}/interrupt` — Interrupt current operation

**Session management:**
- `GET /api/chat/sessions` — List sessions (pagination, `project_slug` filter)
- `GET /api/chat/sessions/{id}` — Get session details
- `GET /api/chat/sessions/{id}/messages` — List message history
- `DELETE /api/chat/sessions/{id}` — Delete session (closes active process)

**WebSocket Event types:**
```
event: assistant_text    → {"content": "..."}
event: thinking          → {"content": "..."}
event: stream_delta      → {"content": "..."} (raw streaming text token)
event: streaming_status  → {"is_streaming": true|false} (stream state change)
event: tool_use          → {"id": "tu_1", "tool": "create_plan", "input": {...}}
event: tool_use_input_resolved → {"id": "tu_1", "tool": "...", "input": {...}} (full input resolved)
event: tool_result       → {"id": "tu_1", "result": {...}, "is_error": false}
event: permission_request → {"id": "pr_1", "tool": "...", "input": {...}}
event: input_request     → {"prompt": "...", "options": [...]}
event: result            → {"session_id": "...", "duration_ms": 5200, "cost_usd": 0.03}
event: error             → {"message": "..."}
```

**MCP Chat Tools (5):**
- `list_chat_sessions` — List sessions with project filter
- `get_chat_session` — Get session details
- `list_chat_messages` — List message history for a session
- `delete_chat_session` — Delete a session
- `chat_send_message` — Send message and wait for complete response (non-streaming)

### Authentication
- `GET /auth/providers` — List available auth providers (password, google, oidc)
- `POST /auth/login` — Password login (root account or Neo4j users)
- `POST /auth/register` — Register a new user account
- `GET /auth/google` — Get Google OAuth authorization URL
- `POST /auth/google/callback` — Exchange auth code for JWT token
- `GET /auth/oidc` — Start generic OIDC login flow
- `GET /auth/oidc/callback` — OIDC code exchange callback
- `GET /auth/me` — Get authenticated user profile (protected)
- `POST /auth/refresh` — Refresh JWT token (protected)

### Events WebSocket
- `GET /ws/events` — CRUD event notifications (entity_types filter, project_id filter)
- `GET /ws/chat/{session_id}` — Chat WebSocket (auth via first message)

### Version Info
- `GET /api/version` — Get server version, build info, and enabled features (public)

### Sync & Watch
- `POST /api/sync` - Manual sync
- `POST /api/watch` - Start auto-sync
- `DELETE /api/watch` - Stop auto-sync

### Meilisearch Maintenance
- `GET /api/meilisearch/stats` - Get code index statistics
- `DELETE /api/meilisearch/orphans` - Delete documents without project_id

## Development Guidelines

1. **Axum 0.8 syntax**: Routes use `{param}` not `:param`
2. **Error handling**: Use `anyhow::Result` and `AppError` for HTTP errors
3. **State**: `ServerState` contains `orchestrator`, `watcher`, and `chat_manager`
4. **Tests**: All API tests require the server running on port 8080
5. **File extensions**: Parser supports 12 languages:
   - Rust: `.rs`
   - TypeScript/JavaScript: `.ts`, `.tsx`, `.js`, `.jsx`
   - Python: `.py`
   - Go: `.go`
   - Java: `.java`
   - C: `.c`, `.h`
   - C++: `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx`
   - Ruby: `.rb`
   - PHP: `.php`
   - Kotlin: `.kt`, `.kts`
   - Swift: `.swift`
   - Bash: `.sh`, `.bash`

## Testing

```bash
# All tests use mock backends (no Docker required for unit tests)
cargo test                     # 752 tests passing
```

## Neo4j Graph Relationships

The knowledge graph uses these relationships:

### Code Structure
- `(Project)-[:CONTAINS]->(File)` - Project contains files
- `(File)-[:CONTAINS]->(Function|Struct|Trait|Enum|Import)` - File contains symbols
- `(File)-[:IMPORTS]->(File)` - File imports another file (resolved from `crate::`, `super::`, `self::`)
- `(Function)-[:CALLS]->(Function)` - Function calls another function
- `(Impl)-[:IMPLEMENTS_FOR]->(Struct|Enum)` - Impl block for a type
- `(Impl)-[:IMPLEMENTS_TRAIT]->(Trait)` - Impl implements a trait (local or external)

### External Traits
External traits (from std, serde, tokio, etc.) are automatically created when:
- A struct/enum uses `#[derive(Debug, Clone, Serialize, ...)]`
- An explicit `impl Trait for Type` references an external trait

External Trait nodes have:
- `is_external: true`
- `source`: The crate name (std, serde, tokio, axum, anyhow, tracing, unknown)

### Plans
- `(Project)-[:HAS_PLAN]->(Plan)` - Project has plans
- `(Plan)-[:HAS_TASK]->(Task)` - Plan contains tasks
- `(Task)-[:HAS_STEP]->(Step)` - Task has steps
- `(Task)-[:DEPENDS_ON]->(Task)` - Task dependencies
- `(Plan)-[:CONSTRAINED_BY]->(Constraint)` - Plan constraints
- `(Task)-[:INFORMED_BY]->(Decision)` - Decisions made during task

### Commits
- `(Task)-[:RESOLVED_BY]->(Commit)` - Commit that resolves a task
- `(Plan)-[:RESULTED_IN]->(Commit)` - Commits generated by a plan

### Releases & Milestones
- `(Project)-[:HAS_RELEASE]->(Release)` - Project has releases
- `(Project)-[:HAS_MILESTONE]->(Milestone)` - Project has milestones
- `(Release)-[:INCLUDES_TASK]->(Task)` - Tasks included in a release
- `(Release)-[:INCLUDES_COMMIT]->(Commit)` - Commits included in a release
- `(Milestone)-[:INCLUDES_TASK]->(Task)` - Tasks included in a milestone

### Chat Sessions
- `(Project)-[:HAS_CHAT_SESSION]->(ChatSession)` - Project has chat sessions

### Workspaces
- `(Project)-[:BELONGS_TO_WORKSPACE]->(Workspace)` - Project is in a workspace
- `(Workspace)-[:HAS_WORKSPACE_MILESTONE]->(WorkspaceMilestone)` - Workspace has cross-project milestones
- `(WorkspaceMilestone)-[:INCLUDES_TASK]->(Task)` - Milestone includes tasks from any project
- `(Workspace)-[:HAS_RESOURCE]->(Resource)` - Workspace has shared resources
- `(Project)-[:IMPLEMENTS_RESOURCE]->(Resource)` - Project is a provider of the resource
- `(Project)-[:USES_RESOURCE]->(Resource)` - Project consumes the resource
- `(Workspace)-[:HAS_COMPONENT]->(Component)` - Workspace has topology components
- `(Component)-[:MAPS_TO_PROJECT]->(Project)` - Component's source code
- `(Component)-[:DEPENDS_ON_COMPONENT {protocol, required}]->(Component)` - Component dependencies

### Knowledge Notes
- `(Note)-[:ATTACHED_TO]->(File|Function|Struct|Trait|Module|Project|Workspace)` - Note is attached to an entity
- `(Note)-[:ATTACHED_TO]->(Task|Plan|Resource|Component)` - Note is attached to planning or workspace entities
- `(Note)-[:SUPERSEDES]->(Note)` - Note replaces an older note
- `(Note)-[:DERIVED_FROM]->(Note)` - Note extends another note

Notes attached to a Workspace automatically propagate to all projects in that workspace with a relevance decay factor (0.8).

Note nodes have:
- `note_type`: guideline, gotcha, pattern, context, tip, observation, assertion
- `status`: active, needs_review, stale, obsolete, archived
- `importance`: critical, high, medium, low
- `staleness_score`: 0.0 - 1.0 (auto-calculated based on time decay)
- `scope_type`, `scope_path`: hierarchical scope (project, module, file, function)

## Meilisearch Indexing

### Code Index (`code`)
- `symbols` - Function/struct/trait names (highest search priority)
- `docstrings` - Documentation strings for semantic search
- `signatures` - Function signatures
- `path` - File path
- `imports` - Import paths
- `project_id`, `project_slug` - Required for project scoping

Note: Full file content is NOT stored in Meilisearch. Use Neo4j for structural queries.

### Notes Index (`notes`)
- `content` - Note text (highest search priority)
- `tags` - Categorization tags
- `scope_path` - Entity scope path
- `anchor_entities` - Linked entities
- `note_type`, `status`, `importance` - Filterable attributes
- `staleness_score`, `created_at` - Sortable attributes

## Common Tasks

### Adding a new API endpoint

1. Add handler in `src/api/handlers.rs` or `src/api/code_handlers.rs`
2. Add route in `src/api/routes.rs`
3. Add test in `tests/api_tests.rs`

### Adding a new language to parser

1. Add tree-sitter grammar to `Cargo.toml`
2. Create extractor in `src/parser/languages/{lang}.rs`
3. Re-export in `src/parser/languages/mod.rs`
4. Add to `SupportedLanguage` enum in `src/parser/mod.rs`
5. Update `from_extension()` and `tree_sitter_language()` methods
6. Add dispatch in `parse_file()` match
7. Add tests in `tests/parser_tests.rs`

### Modifying Neo4j schema

1. Update models in `src/neo4j/models.rs`
2. Update queries in `src/neo4j/client.rs`
3. Add migration if needed (manual Cypher)
