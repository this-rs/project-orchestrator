# Contributing to Project Orchestrator

Thank you for your interest in contributing to Project Orchestrator! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Docker and Docker Compose
- Git

### Setting Up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/this-rs/project-orchestrator.git
   cd project-orchestrator
   ```

2. **Start the backend services**
   ```bash
   docker compose up -d
   ```

3. **Build the project**
   ```bash
   cargo build
   ```

4. **Run tests**
   ```bash
   cargo test
   ```

## Development Workflow

### Code Style

- Run `cargo fmt` before committing to ensure consistent formatting
- Run `cargo clippy -- -D warnings` to catch common mistakes
- Follow Rust naming conventions (snake_case for functions/variables, PascalCase for types)

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting a PR: `cargo test`
- **Unit tests** use mock backends (`MockGraphStore`, `MockSearchStore`) — no external services needed
- **Integration tests** require Neo4j and Meilisearch running
- Current test count: **1992 tests** (unit + integration)

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

Examples:
```
feat(workspace): add multi-project support
fix(neo4j): handle connection timeout gracefully
docs(api): update REST endpoint documentation
test(mcp): add unit tests for workspace handlers
```

## Git Workflow

### Branch naming

| Prefix | Use |
|---|---|
| `docs/<domain>` | documentation and diagrams only for a domain |
| `fix/<domain>-<subject>` | bug fix |
| `test/<domain>-<subject>` | tests only |
| `chore/<subject>` | tooling, CI, maintenance |

### One PR = one vertical slice

A pull request carries a complete slice: the fix, a regression test that
**fails without the fix** (paste the failing command and output in the PR),
the updated diagram under `docs/diagrams/<name>.mmd` (or an explicit
`Diagram-Unchanged: <name> — <reason>`), and the documentation. Diagram-only
PRs describing unverified code are not accepted. The PR template checklist
must be fully ticked; an unticked box means the PR is not mergeable.

Always report coverage as two figures when both exist: **raw** and **gated**
(see `docs/COVERAGE.md`). Never exclude code from coverage to make a number pass.

### Pushing

Never push directly to `main`. Push your branch with an explicit refspec:

```bash
git push origin HEAD:refs/heads/<branch>
```

Do not create releases, tags or version bumps in a feature PR.

### Build directory per worktree

When you use several `git worktree`s, give each one its own build directory so
they never contend for the same Cargo lock or invalidate each other's cache:

```bash
export CARGO_TARGET_DIR=/path/to/target-<worktree-name>
```

### Pre-push hooks

Make sure the Rust toolchain is on the `PATH` before pushing, otherwise the
pre-push hooks cannot find `cargo`:

```bash
PATH=$HOME/.cargo/bin:$PATH git push origin HEAD:refs/heads/<branch>
```
Never bypass the hooks with `--no-verify`.

### Patch coverage gate

`codecov.yml` enforces a patch coverage target (currently 80%) on the lines a
PR adds or modifies. The target lives in a single place (`coverage.status.patch.default.target`)
and is meant to ratchet 80% -> 90% -> 100%.

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** and commit them with clear messages

3. **Ensure CI passes**:
   - `cargo fmt --check`
   - `cargo clippy -- -D warnings`
   - `cargo test`

4. **Push your branch** (see *Pushing* above) and create a Pull Request, filling in the PR template

5. **PR Description** should include:
   - Summary of changes
   - Related issue numbers (if any)
   - Testing instructions

## Project Structure

```
src/
├── api/           # REST API handlers and routes (13 files)
├── auth/          # Authentication: JWT, Google OAuth, middleware
├── chat/          # Chat system: WebSocket, sessions, Claude integration
├── config/        # YAML configuration system with env var overrides
├── events/        # Event bus: CRUD notifications via WebSocket + NATS inter-process sync
├── mcp/           # MCP server and tool definitions (22 mega-tools)
├── neo4j/         # Neo4j client, models, GraphStore trait + mock
├── meilisearch/   # Search client, SearchStore trait + mock
├── parser/        # Tree-sitter code parsing (16 languages)
├── plan/          # Plan and task management
├── notes/         # Knowledge notes system
├── orchestrator/  # Core orchestration logic
└── workspace/     # Multi-project workspace support

tests/
├── api_tests.rs         # REST API integration tests
├── workspace_tests.rs   # Workspace integration tests
├── integration_tests.rs # Database integration tests
└── parser_tests.rs      # Parser unit tests
```

> **Note:** The project has 1992 tests total, including unit tests with mock backends (no external services required).

## Adding New Features

### Adding a New MCP Tool

1. Define the tool in `src/mcp/tools.rs`
2. Implement the handler in `src/mcp/handlers.rs`
3. Register the tool in the appropriate `*_tools()` function
4. Add tests for the new tool

### Adding a New Language Parser

1. Add the tree-sitter grammar to `Cargo.toml`
2. Create an extractor in `src/parser/languages/`
3. Update `SupportedLanguage` enum in `src/parser/mod.rs`
4. Add tests in `tests/parser_tests.rs`

### Adding a New REST Endpoint

1. Add the handler in `src/api/handlers.rs` or the appropriate handler file (`auth_handlers.rs`, `chat_handlers.rs`, `note_handlers.rs`, `workspace_handlers.rs`, etc.)
2. Register the route in `src/api/routes.rs` (mark as public or protected)
3. Add integration tests in `tests/api_tests.rs`
4. If adding an MCP tool, also update `src/mcp/tools.rs` and `src/mcp/handlers.rs`

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Rust version, etc.)
- Relevant logs or error messages

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

If you have questions, feel free to:

- Open a GitHub Discussion
- Check existing issues for similar questions
- Review the documentation in `docs/`

Thank you for contributing!
