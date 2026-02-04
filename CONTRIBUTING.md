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
   git clone https://github.com/your-org/project-orchestrator.git
   cd project-orchestrator
   ```

2. **Start the backend services**
   ```bash
   docker compose up -d neo4j meilisearch
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
- Integration tests require the backend services to be running

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

4. **Push your branch** and create a Pull Request

5. **PR Description** should include:
   - Summary of changes
   - Related issue numbers (if any)
   - Testing instructions

## Project Structure

```
src/
├── api/           # REST API handlers and routes
├── mcp/           # MCP server and tool definitions
├── neo4j/         # Neo4j client and graph models
├── meilisearch/   # Search client and indexing
├── parser/        # Tree-sitter code parsing
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

1. Add the handler in `src/api/handlers.rs` or create a new handler file
2. Register the route in `src/api/routes.rs`
3. Add integration tests in `tests/api_tests.rs`

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
