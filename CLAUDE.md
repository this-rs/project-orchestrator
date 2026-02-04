# CLAUDE.md

This file provides guidance to Claude Code when working on the project-orchestrator skill.

## Project Overview

**Project Orchestrator** is a Rust-based service that coordinates AI coding agents on complex projects. It provides:

- Neo4j graph database for code structure and relationships
- Meilisearch for semantic search across code and decisions
- Tree-sitter for multi-language code parsing
- HTTP API for plans, tasks, decisions, and code exploration
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

## Project Structure

```
src/
├── api/
│   ├── mod.rs           # API module exports
│   ├── routes.rs        # Route definitions (axum)
│   ├── handlers.rs      # Plan/Task/Decision handlers
│   └── code_handlers.rs # Code exploration endpoints
├── neo4j/
│   ├── client.rs        # Neo4j connection and queries
│   └── models.rs        # Graph node types
├── meilisearch/
│   ├── client.rs        # Meilisearch connection
│   └── models.rs        # Search document types
├── parser/
│   ├── mod.rs           # CodeParser, SupportedLanguage, dispatch
│   ├── helpers.rs       # Shared utility functions
│   └── languages/       # Per-language extractors (12 languages)
├── plan/
│   ├── manager.rs       # Plan/Task CRUD operations
│   └── models.rs        # Plan/Task/Decision types
├── orchestrator/
│   ├── runner.rs        # Main orchestrator logic
│   ├── context.rs       # Agent context builder
│   └── watcher.rs       # File watcher for auto-sync
├── lib.rs               # Library exports
└── main.rs              # CLI entry point

tests/
├── api_tests.rs         # HTTP API tests (29)
├── integration_tests.rs # Database tests (8)
└── parser_tests.rs      # Parser tests (33)
```

## Key APIs

### Plans & Tasks
- `POST /api/plans` - Create plan (with optional `project_id` to associate with project)
- `GET /api/plans/{id}` - Get plan details
- `PUT /api/plans/{id}/project` - Link plan to a project
- `DELETE /api/plans/{id}/project` - Unlink plan from project
- `GET /api/plans/{id}/next-task` - Get next available task
- `GET /api/plans/{id}/dependency-graph` - Get task dependency graph for visualization
- `GET /api/plans/{id}/critical-path` - Get longest dependency chain
- `POST /api/plans/{id}/tasks` - Add task (with title, priority, tags, acceptance_criteria, affected_files)
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

## Meilisearch Indexing

The `code` index stores:
- `symbols` - Function/struct/trait names (highest search priority)
- `docstrings` - Documentation strings for semantic search
- `signatures` - Function signatures
- `path` - File path
- `imports` - Import paths
- `project_id`, `project_slug` - Required for project scoping

Note: Full file content is NOT stored in Meilisearch. Use Neo4j for structural queries.

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
