# CLAUDE.md

This file provides guidance to Claude Code when working on the project-orchestrator skill.

## Documentation

For user-facing documentation, see the `docs/` folder:

- **[Installation Guide](docs/setup/installation.md)** — Full setup instructions
- **[Getting Started](docs/guides/getting-started.md)** — Tutorial for new users
- **[API Reference](docs/api/reference.md)** — REST API documentation
- **[MCP Tools](docs/api/mcp-tools.md)** — All 84 MCP tools documented
- **Integration Guides:**
  - [Claude Code](docs/integrations/claude-code.md)
  - [OpenAI Agents](docs/integrations/openai.md)
  - [Cursor](docs/integrations/cursor.md)
- **[Multi-Agent Workflows](docs/guides/multi-agent-workflow.md)** — Advanced coordination
- **[Knowledge Notes](docs/guides/knowledge-notes.md)** — Contextual knowledge capture system

## Project Overview

**Project Orchestrator** is a Rust-based service that coordinates AI coding agents on complex projects. It provides:

- Neo4j graph database for code structure and relationships
- Meilisearch for semantic search across code and decisions
- Tree-sitter for multi-language code parsing
- HTTP API for plans, tasks, decisions, and code exploration
- MCP server for Claude Code integration (62 tools)
- File watcher for auto-syncing changes

## Build Commands

```bash
cargo build --release          # Build release binary
cargo test                     # Run all tests (70 total)
cargo test --test api_tests    # API tests only (29 tests)
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

The MCP server exposes 62 tools organized by category:

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
│   └── mcp-tools.md         # MCP tools reference (84 tools)
└── guides/
    ├── getting-started.md   # Tutorial for new users
    ├── multi-agent-workflow.md # Multi-agent coordination
    └── knowledge-notes.md   # Knowledge Notes system guide

src/
├── api/
│   ├── mod.rs           # API module exports
│   ├── routes.rs        # Route definitions (axum)
│   ├── handlers.rs      # Plan/Task/Decision handlers
│   ├── code_handlers.rs # Code exploration endpoints
│   └── note_handlers.rs # Knowledge Notes endpoints
├── mcp/
│   ├── mod.rs           # MCP module exports
│   ├── protocol.rs      # JSON-RPC 2.0 types
│   ├── tools.rs         # Tool definitions (84 tools)
│   ├── handlers.rs      # Tool implementations
│   └── server.rs        # MCP server (stdio)
├── neo4j/
│   ├── client.rs        # Neo4j connection and queries
│   └── models.rs        # Graph node types
├── meilisearch/
│   ├── client.rs        # Meilisearch connection
│   └── models.rs        # Search document types
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
├── lib.rs               # Library exports
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

### Query Parameters (Pagination & Filtering)

List endpoints (`GET /api/plans`, `GET /api/tasks`, `GET /api/projects/{id}/releases`, `GET /api/projects/{id}/milestones`, `GET /api/projects`) support:

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
3. **State**: `ServerState` contains `orchestrator` and `watcher`
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
# Start server for API tests
./target/release/orchestrator serve &

# Run tests
cargo test

# Expected: 70 tests passing
# - 29 API tests
# - 8 integration tests
# - 33 parser tests
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

### Knowledge Notes
- `(Note)-[:ABOUT]->(File|Function|Struct|Trait|Module|Project)` - Note is about an entity
- `(Note)-[:ABOUT]->(Task|Plan)` - Note is about a task or plan
- `(Note)-[:SUPERSEDES]->(Note)` - Note replaces an older note
- `(Note)-[:DERIVED_FROM]->(Note)` - Note extends another note

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
