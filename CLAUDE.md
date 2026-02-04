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
cargo test                     # Run all tests (60 total)
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
│   └── mod.rs           # Tree-sitter code parser
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
├── api_tests.rs         # HTTP API tests (27)
├── integration_tests.rs # Database tests (7)
└── parser_tests.rs      # Parser tests (17)
```

## Key APIs

### Plans & Tasks
- `POST /api/plans` - Create plan
- `GET /api/plans/{id}` - Get plan details
- `GET /api/plans/{id}/next-task` - Get next available task
- `POST /api/plans/{id}/tasks` - Add task (with title, priority, tags, acceptance_criteria, affected_files)
- `PATCH /api/tasks/{id}` - Update task
- `POST /api/tasks/{id}/decisions` - Record decision

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
5. **File extensions**: Parser supports `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.py`, `.go`

## Testing

```bash
# Start server for API tests
./target/release/orchestrator serve &

# Run tests
cargo test

# Expected: 60 tests passing
# - 29 API tests
# - 8 integration tests
# - 23 parser tests
```

## Neo4j Graph Relationships

The knowledge graph uses these relationships:

### Code Structure
- `(Project)-[:CONTAINS]->(File)` - Project contains files
- `(File)-[:CONTAINS]->(Function|Struct|Trait|Enum|Import)` - File contains symbols
- `(File)-[:IMPORTS]->(File)` - File imports another file (resolved from `crate::`, `super::`, `self::`)
- `(Function)-[:CALLS]->(Function)` - Function calls another function
- `(Impl)-[:IMPLEMENTS_FOR]->(Struct|Enum)` - Impl block for a type
- `(Impl)-[:IMPLEMENTS_TRAIT]->(Trait)` - Impl implements a trait

### Plans
- `(Plan)-[:HAS_TASK]->(Task)` - Plan contains tasks
- `(Task)-[:HAS_STEP]->(Step)` - Task has steps
- `(Task)-[:DEPENDS_ON]->(Task)` - Task dependencies
- `(Plan)-[:CONSTRAINED_BY]->(Constraint)` - Plan constraints
- `(Task)-[:MADE]->(Decision)` - Decisions made during task

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
2. Update `CodeParser::new()` in `src/parser/mod.rs`
3. Add extraction logic in `parse_file()`
4. Add tests in `tests/parser_tests.rs`

### Modifying Neo4j schema

1. Update models in `src/neo4j/models.rs`
2. Update queries in `src/neo4j/client.rs`
3. Add migration if needed (manual Cypher)
